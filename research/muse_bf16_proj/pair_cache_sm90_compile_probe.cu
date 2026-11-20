#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned long long pol_ef() {
  unsigned long long p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(p));
  return p;
}

__device__ __forceinline__ unsigned long long pol_el() {
  unsigned long long p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
               : "=l"(p));
  return p;
}

__device__ __forceinline__ uint4 ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld4_ef(const uint4* p) {
  uint4 r;
  const unsigned long long policy = pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(policy));
  return r;
}

__device__ __forceinline__ uint4 ld4_el(const uint4* p) {
  uint4 r;
  const unsigned long long policy = pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(policy));
  return r;
}

__global__ void probe(const uint4* in, uint4* out) {
  const uint4 a = ld4_na(in + threadIdx.x);
  const uint4 b = ld4_ef(in + blockDim.x + threadIdx.x);
  const uint4 c = ld4_el(in + 2 * blockDim.x + threadIdx.x);
  out[threadIdx.x] = make_uint4(a.x ^ b.x ^ c.x, a.y ^ b.y ^ c.y,
                                a.z ^ b.z ^ c.z, a.w ^ b.w ^ c.w);
}

__device__ __forceinline__ int2 ld2_na(const int2* p) {
  int2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ unsigned short ld1_el(
    const unsigned short* p, unsigned long long policy) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.u16 %0, [%1], %2;"
               : "=h"(r) : "l"(p), "l"(policy));
  return r;
}

__global__ void probe_v271(const int2* table, const unsigned short* x,
                           int2* table_out, unsigned short* x_out) {
  const unsigned long long policy = pol_el();
  table_out[threadIdx.x] = ld2_na(table + threadIdx.x);
  x_out[threadIdx.x] = ld1_el(x + threadIdx.x, policy);
}

__device__ __forceinline__ float probe_esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) -
         __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}

__device__ __forceinline__ uint32_t probe_cg_hib(uint32_t c,
                                                  uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}

// Exact v265 distributed-escape kernel body, with only helper names prefixed
// to keep this standalone translation unit collision-free.
template <int NTH, int K>
__global__ void __launch_bounds__(NTH) v265_pair_kernel(
    const uint8_t* __restrict__ P, const __nv_bfloat16* __restrict__ x,
    const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int ROWS = 2;
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  constexpr int CHUNKS = K / 16;
  static_assert(PER >= 1, "cgemv_pair: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[ROWS][NWARP];
  __shared__ float ered[ROWS][NWARP];
  const int2* tb = etab + (size_t)blockIdx.x * pmax;
  int cnt = (lane == 0) ? __ldg(&tb[0]).x : 0;
  cnt = __shfl_sync(0xffffffffu, cnt, 0);
  float ec[ROWS] = {0.f, 0.f};
  for (int i = 1 + 32 * warp + lane; i <= cnt; i += NTH) {
    const int2 en = __ldg(&tb[i]);
    const float c = __bfloat162float(x[en.x & 0xFFFF]) *
                    probe_esc_delta((uint32_t)en.y, base4 & 0xFFu);
    const int rl = en.x >> 16;
    ec[0] += (rl == 0) ? c : 0.f;
    ec[1] += (rl == 1) ? c : 0.f;
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = ec[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) ered[r][warp] = v;
  }

  uint4 lw[ROWS][PER];
  uint4 cw[PER];
  uint4 xv[PER][2];
  const uint4* pv = reinterpret_cast<const uint4*>(P) +
                    (size_t)blockIdx.x * 3 * CHUNKS;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int q = j * NTH + tid;
    lw[0][j] = __ldg(pv + q);
    lw[1][j] = __ldg(pv + CHUNKS + q);
    cw[j] = __ldg(pv + 2 * CHUNKS + q);
  }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }

  float acc[ROWS] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16);
        xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u);
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {
          lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {
          r == 0 ? cw[j].x : cw[j].z,
          r == 0 ? cw[j].y : cw[j].w};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const uint32_t ha = probe_cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
        const uint32_t hb = probe_cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu,
                                         base4);
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140);
        const uint32_t p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140);
        const uint32_t p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v);
        v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v);
        v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v);
        v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v);
        v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += ered[tid][q];
    out[2 * blockIdx.x + tid] = __float2bfloat16_rn(v);
  }
}

extern "C" void instantiate_v265(
    const uint8_t* p, const __nv_bfloat16* x, const int2* etab, int pmax,
    __nv_bfloat16* out, uint32_t base4, cudaStream_t stream) {
  v265_pair_kernel<256, 8192><<<1, 256, 0, stream>>>(
      p, x, etab, pmax, out, base4);
  v265_pair_kernel<512, 8192><<<1, 512, 0, stream>>>(
      p, x, etab, pmax, out, base4);
  v265_pair_kernel<64, 1024><<<1, 64, 0, stream>>>(
      p, x, etab, pmax, out, base4);
  v265_pair_kernel<32, 1024><<<1, 32, 0, stream>>>(
      p, x, etab, pmax, out, base4);
  v265_pair_kernel<128, 4096><<<1, 128, 0, stream>>>(
      p, x, etab, pmax, out, base4);
  v265_pair_kernel<256, 4096><<<1, 256, 0, stream>>>(
      p, x, etab, pmax, out, base4);
}
