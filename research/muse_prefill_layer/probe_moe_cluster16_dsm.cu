// Isolated SM90a feasibility probe for an on-chip W13 -> SwiGLU -> W2 handoff.
//
// This is deliberately not wired into submission.py.  Sixteen CTAs own one
// 64-token expert tile.  Each CTA forms one 128-column BF16 SwiGLU fragment;
// sixteen ranks therefore publish 2,048 intermediate columns per round.  Every
// CTA consumes all fragments with the production m64n64 WGMMA and exact NVFP4
// register decode while owning 256 W2 output columns (four consumer
// warpgroups).  Two rounds cover the full 4,096-wide intermediate.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

#ifndef PROBE_ROWS
#define PROBE_ROWS 256
#endif

constexpr int kCluster = 16;
constexpr int kRows = PROBE_ROWS;
constexpr int kInterCols = kRows == 256 ? 64 : 128;
constexpr int kW13Rounds = 4096 / (kCluster * kInterCols);
constexpr int kFragmentElems = kRows * kInterCols;
constexpr int kThreads = kRows == 256 ? 384 : 512;
constexpr int kFragmentBytes = kFragmentElems * sizeof(__nv_bfloat16);
constexpr int kLocalBytes = kW13Rounds * kFragmentBytes;
constexpr int kK64Elems = kRows * 64;
constexpr int kK64Bytes = kK64Elems * sizeof(__nv_bfloat16);
constexpr int kWeightStageBytes = kRows == 256 ? 4608 : 9216;
constexpr int kSlotBytes = kK64Bytes + kWeightStageBytes;
constexpr int kRingBytes = 2 * kSlotBytes;
constexpr int kOutputStageBytes = kRows == 256 ? 0 : kRows * 256 * sizeof(__nv_bfloat16);
constexpr int kSmemBytes = kLocalBytes + kRingBytes + kOutputStageBytes + 1024;

__device__ __forceinline__ float silu_fast(float x) {
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar),
               "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar,
                                               uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::
                   "r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::
          "r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src,
                                         uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::
          "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr & 0x3FFFF) >> 4);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}

__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t bf162_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ __nv_bfloat162 u32_bf162(uint32_t v) {
  return *reinterpret_cast<__nv_bfloat162*>(&v);
}

__device__ __forceinline__ uint32_t scale_pair_sel(uint32_t w, int hi,
                                                   uint32_t m16) {
  return __byte_perm(w, 0u, hi ? 0x4342 : 0x4140) * m16 + 0x77807780u;
}

__device__ __forceinline__ void convert8(uint32_t q, uint32_t sb,
                                         uint32_t (&a)[4], uint32_t m16,
                                         uint32_t m64) {
  const __nv_bfloat162 s = u32_bf162(sb);
  const __nv_bfloat162 slo = __low2bfloat162(s);
  const __nv_bfloat162 shi = __high2bfloat162(s);
  constexpr uint32_t M = 0x81C081C0u;
  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) |
                      (__funnelshift_l(q, q, 14) & 0x01000100u);
  a[0] = bf162_u32(__hmul2(u32_bf162(r0), slo));
  a[1] = bf162_u32(__hmul2(u32_bf162(r1), shi));
  a[2] = bf162_u32(__hmul2(u32_bf162(r2), slo));
  a[3] = bf162_u32(__hmul2(u32_bf162(r3), shi));
}

template <int N>
__device__ __forceinline__ void wgmma_rs_probe(float (&d)[N / 2],
                                               const uint32_t (&a)[4],
                                               uint64_t desc, int scale_d);

template <>
__device__ __forceinline__ void wgmma_rs_probe<64>(
    float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
      "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc),
      "r"(scale_d));
}

template <>
__device__ __forceinline__ void wgmma_rs_probe<128>(
    float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
      "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
      "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
      "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
      "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
      "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
      "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
      "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
      "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
      "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc),
      "r"(scale_d));
}

template <>
__device__ __forceinline__ void wgmma_rs_probe<256>(
    float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
      "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
      "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
      "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
      "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
      "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
      "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
      "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
      "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
      "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
      "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
      "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
      "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
      "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
      "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
      "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
      "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
      "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
      "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),
      "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
      "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
      "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
      "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
      "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
      "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
      "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc),
      "r"(scale_d));
}

}  // namespace

#ifndef OKC_WGMMA_HELPERS_ONLY

#if PROBE_ROWS != 256
extern "C" __global__ __launch_bounds__(kThreads, 1)
void moe_cluster16_dsm_probe(const __nv_bfloat16* __restrict__ x,
                             const uint8_t* __restrict__ w13,
                             const uint8_t* __restrict__ w2,
                             __nv_bfloat16* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int tid = static_cast<int>(threadIdx.x);

  extern __shared__ __align__(128) unsigned char raw[];
  __nv_bfloat16* local = reinterpret_cast<__nv_bfloat16*>(raw);
  unsigned char* ring = raw + kLocalBytes;
  __nv_bfloat16* output_stage =
      reinterpret_cast<__nv_bfloat16*>(ring + kRingBytes);
  uint64_t* bar_mem = reinterpret_cast<uint64_t*>(
      reinterpret_cast<unsigned char*>(output_stage) + kOutputStageBytes);
  const uint32_t full0 = smem_u32(bar_mem + 0);
  const uint32_t full1 = smem_u32(bar_mem + 1);

  const uint32_t opaque = static_cast<uint32_t>(clock64() & 0u);
  const uint32_t m16 = 16u + opaque;
  const uint32_t m64 = 64u + opaque;

  // W13: four consumer warpgroups produce 256 gate/up rows per CTA, which
  // become 128 BF16 SwiGLU columns.  Across sixteen CTAs and two rounds this
  // covers all 4,096 intermediate columns.  The accumulator array is scoped
  // to W13 and is dead before W2 starts, so ptxas can reuse the same registers.
  for (int round = 0; round < 2; ++round) {
    float acc[kRows / 2];
#pragma unroll
    for (int i = 0; i < kRows / 2; ++i) acc[i] = 0.0f;

    if (tid == 0) {
      mbar_init(full0, 1);
      mbar_init(full1, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int cluster_id = static_cast<int>(blockIdx.x) / kCluster;
    const int output_tile = round * kCluster + rank;
    if (tid == 0) {
      unsigned char* slot = ring;
      const __nv_bfloat16* xsrc =
          x + static_cast<size_t>(cluster_id) * kRows * 4096;
      const uint8_t* wsrc =
          w13 + static_cast<size_t>(output_tile) * 64 * kWeightStageBytes;
      mbar_expect_tx(full0, kK64Bytes + kWeightStageBytes);
      bulk_g2s(smem_u32(slot), xsrc, kK64Bytes, full0);
      bulk_g2s(smem_u32(slot + kK64Bytes), wsrc, kWeightStageBytes,
               full0);
    }

    for (int stage_index = 0; stage_index < 64; ++stage_index) {
      unsigned char* slot = ring + (stage_index & 1) * kSlotBytes;
      __nv_bfloat16* xstage = reinterpret_cast<__nv_bfloat16*>(slot);
      const uint32_t full = (stage_index & 1) ? full1 : full0;
      mbar_wait(full, (stage_index >> 1) & 1);

      if (tid == 0 && stage_index + 1 < 64) {
        const int next = stage_index + 1;
        unsigned char* next_slot = ring + (next & 1) * kSlotBytes;
        const uint32_t next_full = (next & 1) ? full1 : full0;
        const __nv_bfloat16* xsrc =
            x + (static_cast<size_t>(cluster_id) * kRows * 4096 +
                 next * kK64Elems);
        const uint8_t* wsrc =
            w13 + (static_cast<size_t>(output_tile) * 64 + next) *
                      kWeightStageBytes;
        mbar_expect_tx(next_full, kK64Bytes + kWeightStageBytes);
        bulk_g2s(smem_u32(next_slot), xsrc, kK64Bytes, next_full);
        bulk_g2s(smem_u32(next_slot + kK64Bytes), wsrc,
                 kWeightStageBytes, next_full);
      }

      if (tid < 512) {
        const int wg = tid >> 7;
        const int w = (tid >> 5) & 3;
        const int lane = tid & 31;
        const int g = lane >> 2;
        const int woff = ((wg * 4 + w) * 32 + lane) * 16;
        const int soff = 8192 + ((wg * 4 + w) * 8 + g) * 8;
        const unsigned char* ws = slot + kK64Bytes;
        const uint4 wq = *reinterpret_cast<const uint4*>(ws + woff);
        const uint2 sc = *reinterpret_cast<const uint2*>(ws + soff);
        uint32_t afrag[4][4];
        const uint64_t desc = desc_sw128(smem_u32(xstage));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q =
              j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j], m16,
                   m64);
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_rs_probe<kRows>(acc, afrag[j], desc + 2 * j, 1);
        wg_commit();
        wg_wait<0>();
      }
      __syncthreads();
    }

    if (tid < 512) {
      const int wg = tid >> 7;
      const int w = (tid >> 5) & 3;
      const int lane = tid & 31;
      const int q = wg * 4 + w;
      __nv_bfloat16* dst = local + round * kFragmentElems;
#pragma unroll
      for (int j4 = 0; j4 < kRows / 32; ++j4) {
        uint32_t packed[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int jj = 4 * j4 + u;
          const float gate0 = acc[4 * jj];
          const float gate1 = acc[4 * jj + 1];
          const float up0 = acc[4 * jj + 2];
          const float up1 = acc[4 * jj + 3];
          const __nv_bfloat162 h = __floats2bfloat162_rn(
              silu_fast(gate0) * up0, silu_fast(gate1) * up1);
          packed[u] = bf162_u32(h);
        }
        const int rr = lane & 7;
        const int token_row = 8 * (4 * j4 + (lane >> 3)) + rr;
        const uint32_t addr =
            smem_u32(dst + token_row * kInterCols + 8 * (q ^ rr));
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], "
            "{%1, %2, %3, %4};"
            :: "r"(addr), "r"(packed[0]), "r"(packed[1]),
               "r"(packed[2]), "r"(packed[3])
            : "memory");
      }
    }
    __syncthreads();
  }
  cluster.sync();

  // W2: every CTA consumes the complete distributed 4,096-column
  // intermediate and owns 256 output columns.
  float acc[kRows / 2];
#pragma unroll
  for (int i = 0; i < kRows / 2; ++i) acc[i] = 0.0f;

  if (tid == 0) {
    mbar_init(full0, 1);
    mbar_init(full1, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    const uint8_t* wsrc =
        w2 + static_cast<size_t>(rank) * 64 * kWeightStageBytes;
    mbar_expect_tx(full0, kWeightStageBytes);
    bulk_g2s(smem_u32(ring + kK64Bytes), wsrc, kWeightStageBytes,
             full0);
  }

  for (int round = 0; round < 2; ++round) {
#pragma unroll
    for (int owner = 0; owner < kCluster; ++owner) {
      const __nv_bfloat16* remote_base =
          cluster.map_shared_rank(local, owner);
      const __nv_bfloat16* remote = remote_base + round * kFragmentElems;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int stage_index = (round * kCluster * 2 + owner * 2 + half);
        unsigned char* slot = ring + (stage_index & 1) * kSlotBytes;
        __nv_bfloat16* xstage = reinterpret_cast<__nv_bfloat16*>(slot);
        for (int i = tid; i < kK64Elems; i += kThreads)
          xstage[i] = remote[half * kK64Elems + i];
        __syncthreads();

        const uint32_t full = (stage_index & 1) ? full1 : full0;
        mbar_wait(full, (stage_index >> 1) & 1);
        if (tid == 0 && stage_index + 1 < 64) {
          const int next = stage_index + 1;
          unsigned char* next_slot = ring + (next & 1) * kSlotBytes;
          const uint32_t next_full = (next & 1) ? full1 : full0;
          const uint8_t* wsrc =
              w2 + (static_cast<size_t>(rank) * 64 + next) *
                       kWeightStageBytes;
          mbar_expect_tx(next_full, kWeightStageBytes);
          bulk_g2s(smem_u32(next_slot + kK64Bytes), wsrc,
                   kWeightStageBytes, next_full);
        }

        if (tid < 512) {
          const int wg = tid >> 7;
          const int w = (tid >> 5) & 3;
          const int lane = tid & 31;
          const int g = lane >> 2;
          const int woff = ((wg * 4 + w) * 32 + lane) * 16;
          const int soff = 8192 + ((wg * 4 + w) * 8 + g) * 8;
          const unsigned char* ws = slot + kK64Bytes;
          const uint4 wq = *reinterpret_cast<const uint4*>(ws + woff);
          const uint2 sc = *reinterpret_cast<const uint2*>(ws + soff);
          uint32_t afrag[4][4];
          const uint64_t desc = desc_sw128(smem_u32(xstage));
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t q =
                j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
            const uint32_t sw = j < 2 ? sc.x : sc.y;
            convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j],
                     m16, m64);
          }
          wg_fence();
#pragma unroll
          for (int j = 0; j < 4; ++j)
            wgmma_rs_probe<kRows>(acc, afrag[j], desc + 2 * j, 1);
          wg_commit();
          wg_wait<0>();
        }
        __syncthreads();
      }
    }
  }

  if (tid < 512) {
#pragma unroll
    for (int j = 0; j < kRows / 2; ++j) {
      const int ix = (tid * (kRows / 2) + j) & (kRows * 256 - 1);
      output_stage[ix] = __float2bfloat16_rn(acc[j]);
    }
  }
  __syncthreads();

  for (int i = tid; i < kRows * 256; i += kThreads) {
    sink[(static_cast<size_t>(blockIdx.x) * kRows * 256) + i] = output_stage[i];
  }
#else
  (void)x;
  (void)w13;
  (void)w2;
  (void)sink;
#endif
}

extern "C" cudaError_t launch_moe_cluster16_dsm_probe(
    const __nv_bfloat16* x, const uint8_t* w13, const uint8_t* w2,
    __nv_bfloat16* sink, int clusters, cudaStream_t stream) {
  cudaError_t err = cudaFuncSetAttribute(
      moe_cluster16_dsm_probe, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(
      moe_cluster16_dsm_probe,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess) return err;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kCluster;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(clusters * kCluster, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, moe_cluster16_dsm_probe, x, w13, w2,
                            sink);
}

#else  // PROBE_ROWS == 256

extern "C" __global__ __launch_bounds__(384, 1)
void moe_cluster16_dsm_probe(const __nv_bfloat16* __restrict__ x,
                             const uint8_t* __restrict__ w13,
                             const uint8_t* __restrict__ w2,
                             __nv_bfloat16* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int tid = static_cast<int>(threadIdx.x);
  const bool consumer = tid >= 128;
  const int ctid = tid - 128;

  extern __shared__ __align__(128) unsigned char raw[];
  __nv_bfloat16* local = reinterpret_cast<__nv_bfloat16*>(raw);
  unsigned char* ring = raw + kLocalBytes;
  uint64_t* bar_mem = reinterpret_cast<uint64_t*>(ring + kRingBytes);
  const uint32_t full0 = smem_u32(bar_mem + 0);
  const uint32_t full1 = smem_u32(bar_mem + 1);
  const uint32_t opaque = static_cast<uint32_t>(clock64() & 0u);
  const uint32_t m16 = 16u + opaque;
  const uint32_t m64 = 64u + opaque;
  const int cluster_id = static_cast<int>(blockIdx.x) / kCluster;

  // Four W13 rounds.  Each CTA writes 64 SwiGLU columns, so the cluster
  // publishes 1,024 columns per round and all 4,096 after four rounds.
  for (int round = 0; round < 4; ++round) {
    float acc[128];  // first WGMMA uses scale_d=0; no ordinary register definition
    if (tid == 0) {
      mbar_init(full0, 1);
      mbar_init(full1, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    const int output_tile = round * kCluster + rank;
    if (tid == 0) {
      const __nv_bfloat16* xsrc =
          x + static_cast<size_t>(cluster_id) * kRows * 4096;
      const uint8_t* wsrc =
          w13 + static_cast<size_t>(output_tile) * 64 * kWeightStageBytes;
      mbar_expect_tx(full0, kK64Bytes + kWeightStageBytes);
      bulk_g2s(smem_u32(ring), xsrc, kK64Bytes, full0);
      bulk_g2s(smem_u32(ring + kK64Bytes), wsrc, kWeightStageBytes,
               full0);
    }

    for (int stage_index = 0; stage_index < 64; ++stage_index) {
      unsigned char* slot = ring + (stage_index & 1) * kSlotBytes;
      __nv_bfloat16* xstage = reinterpret_cast<__nv_bfloat16*>(slot);
      const uint32_t full = (stage_index & 1) ? full1 : full0;
      mbar_wait(full, (stage_index >> 1) & 1);

      if (tid == 0 && stage_index + 1 < 64) {
        const int next = stage_index + 1;
        unsigned char* next_slot = ring + (next & 1) * kSlotBytes;
        const uint32_t next_full = (next & 1) ? full1 : full0;
        const __nv_bfloat16* xsrc =
            x + (static_cast<size_t>(cluster_id) * kRows * 4096 +
                 next * kK64Elems);
        const uint8_t* wsrc =
            w13 + (static_cast<size_t>(output_tile) * 64 + next) *
                      kWeightStageBytes;
        mbar_expect_tx(next_full, kK64Bytes + kWeightStageBytes);
        bulk_g2s(smem_u32(next_slot), xsrc, kK64Bytes, next_full);
        bulk_g2s(smem_u32(next_slot + kK64Bytes), wsrc,
                 kWeightStageBytes, next_full);
      }

      if (consumer) {
        const int wg = ctid >> 7;
        const int w = (ctid >> 5) & 3;
        const int lane = ctid & 31;
        const int g = lane >> 2;
        const int woff = ((wg * 4 + w) * 32 + lane) * 16;
        const int soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
        const unsigned char* ws = slot + kK64Bytes;
        const uint4 wq = *reinterpret_cast<const uint4*>(ws + woff);
        const uint2 sc = *reinterpret_cast<const uint2*>(ws + soff);
        uint32_t afrag[4][4];
        const uint64_t desc = desc_sw128(smem_u32(xstage));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q =
              j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j], m16,
                   m64);
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_rs_probe<256>(acc, afrag[j], desc + 2 * j,
                              (stage_index == 0 && j == 0) ? 0 : 1);
        wg_commit();
        wg_wait<0>();
      }
      __syncthreads();
    }

    if (consumer) {
      const int wg = ctid >> 7;
      const int w = (ctid >> 5) & 3;
      const int lane = ctid & 31;
      const int q = wg * 4 + w;
      __nv_bfloat16* dst = local + round * kFragmentElems;
#pragma unroll
      for (int j4 = 0; j4 < 8; ++j4) {
        uint32_t packed[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int jj = 4 * j4 + u;
          const __nv_bfloat162 h = __floats2bfloat162_rn(
              silu_fast(acc[4 * jj]) * acc[4 * jj + 2],
              silu_fast(acc[4 * jj + 1]) * acc[4 * jj + 3]);
          packed[u] = bf162_u32(h);
        }
        const int rr = lane & 7;
        const int token_row = 8 * (4 * j4 + (lane >> 3)) + rr;
        const uint32_t addr =
            smem_u32(dst + token_row * kInterCols + 8 * (q ^ rr));
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], "
            "{%1, %2, %3, %4};"
            :: "r"(addr), "r"(packed[0]), "r"(packed[1]),
               "r"(packed[2]), "r"(packed[3])
            : "memory");
      }
    }
    __syncthreads();
  }
  cluster.sync();

  // Two W2 output phases.  A phase owns 128 columns per CTA and therefore
  // 2,048 columns per cluster.  The four distributed W13 rounds provide the
  // 64 K64 stages consumed below without any global intermediate.
  for (int phase = 0; phase < 2; ++phase) {
    float acc[128];  // first WGMMA uses scale_d=0; no ordinary register definition
    if (tid == 0) {
      mbar_init(full0, 1);
      mbar_init(full1, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int output_tile = phase * kCluster + rank;
    if (tid == 0) {
      const uint8_t* wsrc =
          w2 + static_cast<size_t>(output_tile) * 64 * kWeightStageBytes;
      mbar_expect_tx(full0, kWeightStageBytes);
      bulk_g2s(smem_u32(ring + kK64Bytes), wsrc, kWeightStageBytes,
               full0);
    }

    for (int round = 0; round < 4; ++round) {
#pragma unroll
      for (int owner = 0; owner < kCluster; ++owner) {
        const int stage_index = round * kCluster + owner;
        unsigned char* slot = ring + (stage_index & 1) * kSlotBytes;
        __nv_bfloat16* xstage = reinterpret_cast<__nv_bfloat16*>(slot);
        const __nv_bfloat16* remote_base =
            cluster.map_shared_rank(local, owner);
        const __nv_bfloat16* remote =
            remote_base + round * kFragmentElems;
        for (int i = tid; i < kK64Elems; i += 384) xstage[i] = remote[i];
        __syncthreads();

        const uint32_t full = (stage_index & 1) ? full1 : full0;
        mbar_wait(full, (stage_index >> 1) & 1);
        if (tid == 0 && stage_index + 1 < 64) {
          const int next = stage_index + 1;
          unsigned char* next_slot = ring + (next & 1) * kSlotBytes;
          const uint32_t next_full = (next & 1) ? full1 : full0;
          const uint8_t* wsrc =
              w2 + (static_cast<size_t>(output_tile) * 64 + next) *
                       kWeightStageBytes;
          mbar_expect_tx(next_full, kWeightStageBytes);
          bulk_g2s(smem_u32(next_slot + kK64Bytes), wsrc,
                   kWeightStageBytes, next_full);
        }

        if (consumer) {
          const int wg = ctid >> 7;
          const int w = (ctid >> 5) & 3;
          const int lane = ctid & 31;
          const int g = lane >> 2;
          const int woff = ((wg * 4 + w) * 32 + lane) * 16;
          const int soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
          const unsigned char* ws = slot + kK64Bytes;
          const uint4 wq = *reinterpret_cast<const uint4*>(ws + woff);
          const uint2 sc = *reinterpret_cast<const uint2*>(ws + soff);
          uint32_t afrag[4][4];
          const uint64_t desc = desc_sw128(smem_u32(xstage));
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t q =
                j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
            const uint32_t sw = j < 2 ? sc.x : sc.y;
            convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j], m16,
                     m64);
          }
          wg_fence();
#pragma unroll
          for (int j = 0; j < 4; ++j)
            wgmma_rs_probe<256>(acc, afrag[j], desc + 2 * j,
                                (stage_index == 0 && j == 0) ? 0 : 1);
          wg_commit();
          wg_wait<0>();
        }
        __syncthreads();
      }
    }

    // The production candidate will reuse the now-dead 73 KB ring for the
    // existing two-half stmatrix epilogue.  Direct stores here keep all 128
    // accumulators live for the resource proof without allocating more smem.
    if (consumer) {
      const size_t base =
          (static_cast<size_t>(blockIdx.x) * 2 + phase) * 256 * 128;
#pragma unroll
      for (int i = 0; i < 128; ++i)
        sink[base + ((static_cast<size_t>(ctid) * 128 + i) &
                     (256 * 128 - 1))] = __float2bfloat16_rn(acc[i]);
    }
    __syncthreads();
  }
#else
  (void)x;
  (void)w13;
  (void)w2;
  (void)sink;
#endif
}

extern "C" cudaError_t launch_moe_cluster16_dsm_probe(
    const __nv_bfloat16* x, const uint8_t* w13, const uint8_t* w2,
    __nv_bfloat16* sink, int clusters, cudaStream_t stream) {
  cudaError_t err = cudaFuncSetAttribute(
      moe_cluster16_dsm_probe, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemBytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(
      moe_cluster16_dsm_probe,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess) return err;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kCluster;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(clusters * kCluster, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, moe_cluster16_dsm_probe, x, w13, w2,
                            sink);
}

#endif  // PROBE_ROWS

#endif  // OKC_WGMMA_HELPERS_ONLY
