// Isolated SM90a resource proof for a 16-CTA O-projection -> first-boundary
// fusion.  This file is never imported by submission.py.
//
// The included probe supplies the already-assembled m64n256 register-source
// WGMMA helper and the Cluster16 launch pattern.  This kernel models the
// production topology: two consumer warpgroups per CTA produce 128 rows x 256
// columns, a pair of 16-CTA clusters covers all 8192 columns, the rounded BF16
// projection remains in DSM, and three paired-cluster reductions produce the
// residual/RMS/FP8-pooler outputs.  Pairing preserves the standalone m128n256
// mainloop's reuse of each weight tile across two 64-row consumer WGs.
#define OKC_WGMMA_HELPERS_ONLY 1
#include "probe_moe_cluster16_dsm.cu"

namespace {

constexpr int kOCluster = 16;
constexpr int kORows = 128;
constexpr int kOColsPerCta = 256;
constexpr int kOThreads = 512;  // four consumer WGs; one elected lane issues TMA
constexpr int kOElems = kORows * kOColsPerCta;
constexpr int kORingBytes = 128 * 1024;  // two 64-KiB B slots; reused for FP32 residual
constexpr int kOProjBytes = kOElems * sizeof(__nv_bfloat16);
constexpr int kOSumBytes = 4 * kORows * sizeof(float);
constexpr int kOSmemBytes = kORingBytes + kOProjBytes + kOSumBytes + 256;

__device__ __forceinline__ uint16_t oproj_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// A true output-only scale-D-zero first instruction.  Using the generic +f
// helper for this instruction leaves the accumulator definition ambiguous to
// ptxas; zeroing 128 ordinary registers instead exceeds the 168-register
// WGMMA scheduling budget.  This is the same split used by the earlier clean
// standalone O-projection mainloop.
__device__ __forceinline__ void oproj_wgmma256_init(
    float (&d)[128], uint64_t adesc, uint64_t bdesc) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
    "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
    "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
    "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
    "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
    "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
    "%128, %129, 0, 1, 1, 0, 0;"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
      "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
      "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
      "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
      "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
      "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
      "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
      "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]),
      "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
      "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
      "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]),
      "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]),
      "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]),
      "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
      "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
      "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]),
      "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]),
      "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]),
      "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]),
      "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]),
      "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]),
      "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]),
      "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]),
      "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]),
      "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]),
      "=f"(d[100]), "=f"(d[101]), "=f"(d[102]), "=f"(d[103]),
      "=f"(d[104]), "=f"(d[105]), "=f"(d[106]), "=f"(d[107]),
      "=f"(d[108]), "=f"(d[109]), "=f"(d[110]), "=f"(d[111]),
      "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]),
      "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]),
      "=f"(d[120]), "=f"(d[121]), "=f"(d[122]), "=f"(d[123]),
      "=f"(d[124]), "=f"(d[125]), "=f"(d[126]), "=f"(d[127])
    : "l"(adesc), "l"(bdesc));
}

__device__ __forceinline__ void oproj_wgmma256_accum(
    float (&d)[128], uint64_t adesc, uint64_t bdesc) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
    "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
    "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
    "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
    "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
    "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
    "%128, %129, 1, 1, 1, 0, 0;"
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
    : "l"(adesc), "l"(bdesc));
}

__device__ __forceinline__ void oproj_wgmma128_init(
    float (&d)[64], uint64_t adesc, uint64_t bdesc) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
    "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
    "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
    "%64, %65, 0, 1, 1, 0, 0;"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
      "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
      "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
      "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
      "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
      "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
      "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
      "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]),
      "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
      "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
      "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]),
      "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]),
      "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]),
      "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
      "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
      "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
    : "l"(adesc), "l"(bdesc));
}

__device__ __forceinline__ void oproj_wgmma128_accum(
    float (&d)[64], uint64_t adesc, uint64_t bdesc) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
    "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
    "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
    "%64, %65, 1, 1, 1, 0, 0;"
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
    : "l"(adesc), "l"(bdesc));
}

// Keep the 128-float accumulator in a dedicated allocation scope.  The caller
// enters the cluster epilogue only after this routine has rounded and stored
// the projection to DSM.
__device__ __forceinline__ void oproj_mainloop_ss(
    unsigned char* ring, __nv_bfloat16* proj, int ctid, int wg) {
  float acc[64];
  const int mg = wg >> 1;
  const int ng = wg & 1;
  {
    wg_fence();
    const uint64_t adesc = desc_sw128(smem_u32(ring + mg * 8192));
    const uint64_t bdesc = desc_sw128(smem_u32(ring + 16384 + ng * 16384));
    oproj_wgmma128_init(acc, adesc, bdesc);
#pragma unroll
    for (int j = 1; j < 4; ++j)
      oproj_wgmma128_accum(acc, adesc + 2 * j, bdesc + 2 * j);
    wg_commit();
    wg_wait<1>();
  }
  for (int ks = 1; ks < 64; ++ks) {
    wg_fence();
    const uint32_t bs = smem_u32(ring + (ks & 1) * 65536);
    const uint64_t adesc = desc_sw128(bs + mg * 8192);
    const uint64_t bdesc = desc_sw128(bs + 16384 + ng * 16384);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      oproj_wgmma128_accum(acc, adesc + 2 * j, bdesc + 2 * j);
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 64; ++i)
    proj[ctid * 64 + i] = __float2bfloat16_rn(acc[i]);
}

extern "C" __global__ __launch_bounds__(kOThreads, 1)
void oproj_boundary_cluster16_probe(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ stream,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const __nv_bfloat16* __restrict__ gpre,
    __nv_bfloat16* __restrict__ hidden,
    uint8_t* __restrict__ hidden8,
    float* __restrict__ hidden_scale,
    float* __restrict__ row_rms,
    float* __restrict__ pair_partials,
    int* __restrict__ pair_counters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x) / kOCluster;
  const int pair_id = cluster_index >> 1;
  const int half = cluster_index & 1;
  const int tid = static_cast<int>(threadIdx.x);
  const bool consumer = tid < 512;
  const int wg = tid >> 7;

  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* ring = raw;
  __nv_bfloat16* proj = reinterpret_cast<__nv_bfloat16*>(raw + kORingBytes);
  float* proj_sum = reinterpret_cast<float*>(raw + kORingBytes + kOProjBytes);
  float* proj_rms = proj_sum + kORows;
  float* residual_sum = proj_rms + kORows;
  float* residual_rms = residual_sum + kORows;

  // Resource-faithful two-slot B ring.  The resource probe does not execute on
  // the local GPU, so a cooperative copy is enough; a production candidate
  // would use two-stage 2-D TMA and multicast the common X tile.
  for (int off = tid * 16; off < kORingBytes; off += kOThreads * 16) {
    const size_t wi = (static_cast<size_t>(half * kOCluster + rank) * 32768 + off) &
                      (static_cast<size_t>(8192) * 1024 * 2 - 16);
    *reinterpret_cast<uint4*>(ring + off) =
        *reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(weight) + wi);
  }
  __syncthreads();

  if (consumer) oproj_mainloop_ss(ring, proj, tid, wg);
  __syncthreads();

  // First reduction: exact boundary input is the BF16-rounded projection.
  if (tid < kORows) {
    float ss = 0.0f;
#pragma unroll 4
    for (int c = 0; c < kOColsPerCta; ++c) {
      const float v = __bfloat162float(proj[tid * kOColsPerCta + c]);
      ss = fmaf(v, v, ss);
    }
    proj_sum[tid] = ss;
  }
  __syncthreads();
  cluster.sync();

  if (tid < kORows) {
    float ss = 0.0f;
#pragma unroll
    for (int owner = 0; owner < kOCluster; ++owner) {
      const float* remote = cluster.map_shared_rank(proj_sum, owner);
      ss += remote[tid];
    }
    proj_rms[tid] = ss;  // this cluster's 4096-column half
  }
  __syncthreads();

  // Rank zero of each cluster exchanges one 128-float partial with the paired
  // cluster.  A production persistent kernel resets the three work slots after
  // both halves have copied each result locally; reset traffic is omitted from
  // this resource proof because it is outside the WGMMA live range.
  if (rank == 0) {
    const int slot = pair_id * 3;
    if (tid < kORows)
      pair_partials[((slot * 2 + half) * kORows) + tid] = proj_rms[tid];
    __syncthreads();
    if (tid == 0) { __threadfence(); atomicAdd(pair_counters + slot, 1); }
    __syncthreads();
    while (atomicAdd(pair_counters + slot, 0) < 2) {}
    if (tid < kORows) {
      const float a = pair_partials[((slot * 2 + 0) * kORows) + tid];
      const float b = pair_partials[((slot * 2 + 1) * kORows) + tid];
      proj_rms[tid] = rsqrtf((a + b) * (1.0f / 8192.0f) + 1e-8f);
    }
  }
  cluster.sync();
  const float* full_proj_rms = cluster.map_shared_rank(proj_rms, 0);

  // The dead 128-KiB GEMM ring becomes the FP32 residual staging tile.
  float* residual = reinterpret_cast<float*>(ring);
  if (tid < kORows) {
    const int global_row = pair_id * kORows + tid;
    float ss = 0.0f;
#pragma unroll 4
    for (int c = 0; c < kOColsPerCta; ++c) {
      const int col = half * 4096 + rank * kOColsPerCta + c;
      const int li = tid * kOColsPerCta + c;
      const size_t gi = static_cast<size_t>(global_row) * 8192 + col;
      const float a = __bfloat162float(proj[li]);
      const float s = fmaf(alpha[col], stream[gi],
                           beta[col] * (a * full_proj_rms[tid]));
      residual[li] = s;
      ss = fmaf(s, s, ss);
    }
    residual_sum[tid] = ss;
  }
  __syncthreads();
  cluster.sync();

  if (tid < kORows) {
    float ss = 0.0f;
#pragma unroll
    for (int owner = 0; owner < kOCluster; ++owner) {
      const float* remote = cluster.map_shared_rank(residual_sum, owner);
      ss += remote[tid];
    }
    residual_rms[tid] = ss;
  }
  __syncthreads();

  if (rank == 0) {
    const int slot = pair_id * 3 + 1;
    if (tid < kORows)
      pair_partials[((slot * 2 + half) * kORows) + tid] = residual_rms[tid];
    __syncthreads();
    if (tid == 0) { __threadfence(); atomicAdd(pair_counters + slot, 1); }
    __syncthreads();
    while (atomicAdd(pair_counters + slot, 0) < 2) {}
    if (tid < kORows) {
      const float a = pair_partials[((slot * 2 + 0) * kORows) + tid];
      const float b = pair_partials[((slot * 2 + 1) * kORows) + tid];
      residual_rms[tid] = rsqrtf((a + b) * (1.0f / 8192.0f) + 1e-5f);
      if (half == 0) row_rms[pair_id * kORows + tid] = 1.0f / residual_rms[tid];
    }
  }
  cluster.sync();
  const float* full_residual_rms = cluster.map_shared_rank(residual_rms, 0);

  // Form the exact BF16 hidden value and retain it in the old projection DSM
  // tile until the FP8 row scale is known.
  if (tid < kORows) {
    const int global_row = pair_id * kORows + tid;
    float mx = 0.0f;
#pragma unroll 4
    for (int c = 0; c < kOColsPerCta; ++c) {
      const int col = half * 4096 + rank * kOColsPerCta + c;
      const int li = tid * kOColsPerCta + c;
      const size_t gi = static_cast<size_t>(global_row) * 8192 + col;
      const float gain = 1.0f + __bfloat162float(gpre[col]);
      const __nv_bfloat16 h = __float2bfloat16_rn(
          residual[li] * full_residual_rms[tid] * gain);
      proj[li] = h;
      hidden[gi] = h;
      mx = fmaxf(mx, fabsf(__bfloat162float(h)));
    }
    proj_sum[tid] = mx;
  }
  __syncthreads();
  cluster.sync();

  if (tid < kORows) {
    float mx = 0.0f;
#pragma unroll
    for (int owner = 0; owner < kOCluster; ++owner) {
      const float* remote = cluster.map_shared_rank(proj_sum, owner);
      mx = fmaxf(mx, remote[tid]);
    }
    proj_rms[tid] = mx;
  }
  __syncthreads();

  if (rank == 0) {
    const int slot = pair_id * 3 + 2;
    if (tid < kORows)
      pair_partials[((slot * 2 + half) * kORows) + tid] = proj_rms[tid];
    __syncthreads();
    if (tid == 0) { __threadfence(); atomicAdd(pair_counters + slot, 1); }
    __syncthreads();
    while (atomicAdd(pair_counters + slot, 0) < 2) {}
    if (tid < kORows) {
      const float a = pair_partials[((slot * 2 + 0) * kORows) + tid];
      const float b = pair_partials[((slot * 2 + 1) * kORows) + tid];
      proj_rms[tid] = fmaxf(fmaxf(a, b), 1e-30f) * (1.0f / 448.0f);
      if (half == 0) hidden_scale[pair_id * kORows + tid] = proj_rms[tid];
    }
  }
  cluster.sync();
  const float* full_qscale = cluster.map_shared_rank(proj_rms, 0);

  // Packed FP8 output consumed by the retained Pooler.
  for (int p = tid; p < kOElems / 2; p += kOThreads) {
    const int row = (2 * p) / kOColsPerCta;
    const int col = (2 * p) - row * kOColsPerCta;
    const int global_row = pair_id * kORows + row;
    const size_t gi = static_cast<size_t>(global_row) * 8192 +
                      half * 4096 + rank * kOColsPerCta + col;
    const float inv = 1.0f / full_qscale[row];
    const float lo = __bfloat162float(proj[2 * p]) * inv;
    const float hi = __bfloat162float(proj[2 * p + 1]) * inv;
    *reinterpret_cast<uint16_t*>(hidden8 + gi) = oproj_e4m3x2(lo, hi);
  }
#else
  (void)x; (void)weight; (void)stream; (void)alpha; (void)beta; (void)gpre;
  (void)hidden; (void)hidden8; (void)hidden_scale; (void)row_rms;
  (void)pair_partials; (void)pair_counters;
#endif
}

}  // namespace

extern "C" cudaError_t launch_oproj_boundary_cluster16_probe(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const float* stream, const float* alpha, const float* beta,
    const __nv_bfloat16* gpre, __nv_bfloat16* hidden, uint8_t* hidden8,
    float* hidden_scale, float* row_rms, float* pair_partials,
    int* pair_counters, int clusters, cudaStream_t stream_) {
  cudaError_t err = cudaFuncSetAttribute(
      oproj_boundary_cluster16_probe,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kOSmemBytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(
      oproj_boundary_cluster16_probe,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess) return err;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kOCluster;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(clusters * kOCluster, 1, 1);
  cfg.blockDim = dim3(kOThreads, 1, 1);
  cfg.dynamicSmemBytes = kOSmemBytes;
  cfg.stream = stream_;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, oproj_boundary_cluster16_probe, x, weight,
                            stream, alpha, beta, gpre, hidden, hidden8,
                            hidden_scale, row_rms, pair_partials, pair_counters);
}
