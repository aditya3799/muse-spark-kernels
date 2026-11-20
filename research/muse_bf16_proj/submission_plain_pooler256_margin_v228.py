# okc: problem=muse_bf16_proj
# Ported from team vjeux's Open Kernel Challenge entry: https://github.com/vjeux/okc-2026 (muse_bf16_proj.py),
# licensed under the Apache License, Version 2.0; used and redistributed under its terms. Credit: team vjeux.
# Modifications by team rohanarun: this header, extra autotuner candidates (rq_*), and a per-shape composition block at the end.
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

out[m, n] = x[m, k] @ W[n, k]^T.  For m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 A-fragment order (prepared once), and multiplied
against x (staged in shared memory) on BF16 tensor cores with FP32 accumulation.

Lossless 12-bit weights (BF16 projections): each BF16 weight is stored as its low byte
(exponent bit 0 + 7 mantissa bits, raw) and a 4-bit code for its high byte (sign + exponent
bits 7..1, as an offset into an 8-value window chosen per matrix).  The kernel rebuilds the
exact BF16 bits in registers (a few integer ops per weight).  The rare weights whose exponent
falls outside the window (tiny magnitudes) are exact-corrected: the kernel adds
x[:, k] * (w - w_decoded) for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.

Split-K across warps (shared-memory reduction) and across CTAs (FP32 partials + a
self-resetting semaphore; the last CTA of a row block reduces) keeps ~128 CTAs streaming on
every shape.  The router's FP32 weights run as a BF16 hi + lo pair (concatenated along K)
with FP32 accumulation, as SGLang does above 8 tokens.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
import sys

from torch.utils.cpp_extension import load_inline
import os as _os
_SK2_DELAY = int(_os.environ.get("SK2_DELAY_ESC", "0"))   # debug only: one random escape warp per CTA stalls this many ns at a random stage; ring waits trap after 2 s


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
@@BODY@@
"""

MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
#else
  mbar_wait(bar, parity);
#endif
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
#endif

"""

WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def wgmma_defs(bns):
    """Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
// on sm_80 all 32 producer lanes issue 16-byte cp.async and arrive with cp.async.mbarrier.arrive).
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  // dg (diagnostics only, 0 in production): 1 = skip epilogue, 2 = also skip x / escape-table loads,
  // 3 = also skip all compute (pure weight streaming through the pipeline)
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
#endif
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
#if __CUDA_ARCH__ >= 900
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
#else
    const int st0 = 0;
#endif
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
#if __CUDA_ARCH__ >= 900
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
#else
      for (int off = lane * 16; off < SLOT; off += 512)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + off), "l"(src + off) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
#if __CUDA_ARCH__ >= 900
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
#endif
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;   // keep the loads
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
#else
      // sm_80 check path: same data layout, mma.sync with B fragments read from the swizzled x tile
      const int g8 = lane >> 2, t8 = lane & 3;
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;
        const unsigned char* tile = reinterpret_cast<const unsigned char*>(xs) + (kl >> 6) * TOK * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int jj = 0; jj < MT; ++jj) {
            const int tok = jj * 8 + g8;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t8);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t8);
            mma_bf16(wacc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
        }
      }
#endif
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
#if __CUDA_ARCH__ >= 900
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }
#endif

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;   // (the producer warp has already returned)
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)   // A tile pre-swizzled and contiguous: [rb][kb][BM x 128 B]
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (true) {
      int v = *((volatile int*)(counters + rb));
      if (v - target >= 0) break;
      __nanosleep(32);
    }
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


SK2_SRC = r"""

#ifndef SK2_DELAY_ESC
#define SK2_DELAY_ESC 0
#endif
#if SK2_DELAY_ESC
// debug builds only: every ring wait of the escape and producer warps gives up after 2 s (globaltimer) with a trap, so
// a wedged ring aborts the launch instead of hanging the GPU
__device__ __forceinline__ void sk2_wait_dbg(uint32_t bar, uint32_t parity) {
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  uint32_t ok = 0;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(1000u) : "memory");
    if (ok) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > 2000000000ull) {
      printf("sk2 ring wedged: block %d warp %d bar 0x%x parity %u\n", (int)blockIdx.x, (int)(threadIdx.x >> 5), bar, parity);
      __trap();
    }
  }
}
#define SK2_WAIT(bar, par) sk2_wait_dbg((bar), (par))
// one (escape warp, stage) per CTA per launch stalls SK2_DELAY_ESC ns at the end of its stage step; the pair comes from a
// hash of the launch's start time (read once per CTA by every escape warp from the same smem word) and the block index
#define SK2_DELAY_HOOK(st_) do { if ((st_) == dly_st && ew == dly_w) { const unsigned long long t0_ = gtime(); \
    while (gtime() - t0_ < (unsigned long long)SK2_DELAY_ESC) __nanosleep(500); } } while (0)
#define SK2_DLY_SEED_DECL __shared__ unsigned long long sk2_dly_seed;
#define SK2_DLY_SEED_SET() (sk2_dly_seed = gtime())
#define SK2_DLY_PICK() \
    unsigned long long h_ = sk2_dly_seed ^ (0x9E3779B97F4A7C15ull * (unsigned long long)(blockIdx.x + 1)); \
    h_ ^= h_ >> 30; h_ *= 0xBF58476D1CE4E5B9ull; h_ ^= h_ >> 27; h_ *= 0x94D049BB133111EBull; h_ ^= h_ >> 31; \
    const int dly_w = (int)(h_ % (unsigned long long)NESC), dly_st = (int)((h_ >> 32) % (unsigned long long)(nst > 0 ? nst : 1));
#else
#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DELAY_HOOK(st_) do { } while (0)
#define SK2_DLY_SEED_DECL
#define SK2_DLY_SEED_SET() ((void)0)
#define SK2_DLY_PICK()
#endif
#ifndef SK2_NESC
#define SK2_NESC 3
#endif
#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
#if __CUDA_ARCH__ >= 900
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  SK2_DLY_SEED_DECL
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    SK2_DLY_SEED_SET();
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    SK2_DLY_PICK()
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          SK2_DELAY_HOOK(st);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        SK2_DELAY_HOOK(st);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
      if (dgv & 8) {   // diagnostic: replace the tensor-core instructions by ALU work
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {   // every CTA: SM id, start and end time
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
#endif
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapA, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NWG);
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
#if __CUDA_ARCH__ >= 900
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    if constexpr (HLW) {   // router: BF16 hi and lo tiles [64 rows x 64 k] by TMA (128B swizzle) from [hi | lo]
      tma_load_2d_ef(st + B_BYTES, &mapA, full0 + 8 * slot, (u0 + i) * 64, rb * 64);
      tma_load_2d_ef(st + B_BYTES + 8192, &mapA, full0 + 8 * slot, K + (u0 + i) * 64, rb * 64);
    } else {
      bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
    }
#else
    for (int idx = lane; idx < TOK * 8; idx += 32) {
      const int t = idx >> 3, c = idx & 7;
      const uint32_t dst = st + t * 128 + ((c ^ (t & 7)) << 4);
      const __nv_bfloat16* src = x + (size_t)(t0z + (t < Mz ? t : 0)) * K + (u0 + i) * 64 + c * 8;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < Mz ? 16 : 0) : "memory");
    }
    const char* wa = reinterpret_cast<const char*>(wsrc) + (size_t)i * A_BYTES;
    for (int off = lane * 16; off < A_BYTES; off += 512)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(st + B_BYTES + off), "l"(wa + off) : "memory");
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
  };
#if __CUDA_ARCH__ >= 900
  const bool producer = tid == 0;
#else
  const bool producer = warp == 0;
#endif
#if __CUDA_ARCH__ >= 900
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    if constexpr (HLW) asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
#else
  __syncthreads();
  if (producer)
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
#endif
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#if __CUDA_ARCH__ >= 900
      // router: SS-wgmma on the TMA'd hi and lo tiles (no register-sourced operands: the wgmma pipeline stays full)
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + j * 32), make_desc_sw128(st + j * 32));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + 8192 + j * 32), make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
#else
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // h = 0: hi tile, 1: lo tile (same B) -> FP32 sum of both
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#if __CUDA_ARCH__ >= 900
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
          const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
          for (int jj = 0; jj < TOK / 8; ++jj) {
            const int tok = jj * 8 + g;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
            mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
#endif
        }
      }
#endif
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
      const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const int tok = jj * 8 + g;
        const unsigned char* row = tile + tok * 128;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
        mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
      }
#endif
    }
    }  // !HLW
#if __CUDA_ARCH__ >= 900
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
#else
    named_bar(1, NT);   // check path: every warp is done with this slot -> refill it
    if (producer && i + STAGES < nst) issue(i + STAGES);
#endif
  }
#if __CUDA_ARCH__ >= 900
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
#endif

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  named_bar(1, NT);
  if (tid == 0) {
    // release: every thread's partial (ordered before the CTA barrier) is visible before the count; acquire on the
    // spin makes the other CTAs' partials visible to this CTA after the barrier below
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(counters + zh * nrb + rb) : "memory");
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(counters + zh * nrb + rb) : "memory");
      } while (v - target < 0);
    } else if (s_last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB, mA;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  mA = mB;
  if (HLW && major >= 9) mA = make_map_2d(Wst.data_ptr(), (uint64_t)Wst.size(0), (uint64_t)Wst.size(1), 64);   // [hi | lo]
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, mA, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

# (layer, mt) -> (splits, RG, WS); grid = (N / (16 RG), splits) ~ 128 CTAs
CONFIGS = {
    ("pooler", 1): (1, 2, 8), ("pooler", 2): (2, 4, 4), ("pooler", 4): (4, 8, 2), ("pooler", 8): (16, 8, 1),
    ("unpooler", 1): (2, 1, 8), ("unpooler", 2): (1, 4, 4), ("unpooler", 4): (2, 8, 2), ("unpooler", 8): (8, 8, 1),
    ("qkv_gate", 1): (8, 9, 2), ("qkv_gate", 2): (8, 9, 2), ("qkv_gate", 4): (8, 9, 1), ("qkv_gate", 8): (16, 9, 1),
    ("o", 1): (1, 4, 4), ("o", 2): (1, 4, 4), ("o", 4): (1, 4, 4), ("o", 8): (2, 8, 1),
    ("router", 1): (8, 1, 8), ("router", 2): (8, 1, 8), ("router", 4): (8, 1, 8), ("router", 8): (16, 1, 4),
}
MAX_SKINNY_M = 32
w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}
D_PLAIN = 16
D_COMP = 12
PF = 0

_ext = None


def _dispatch_src():
    return ""  # LDG skinny kernel unused in v6 (TMA path); keep the template uninstantiated
    lines = []
    for (layer, mt), (splits, rg, wsn) in sorted(CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            d = D_COMP if comp else (D_PLAIN if rg * wsn <= 9 and mt <= 4 else 8)
            lines.append(f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch<{mt}, {rg}, {wsn}, {d}, {str(bool(comp)).lower()}, {PF}, {ot}>(Wp, x, out, ws, counters, "
                         f"poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    return "\n  ".join(dict.fromkeys(lines))


MID_CONFIGS = {  # (layer, bn) -> (nwg, splits); grid = (N / (64 nwg), splits) <= 132 CTAs (co-resident)
    ("pooler", 128): (2, 4), ("unpooler", 128): (2, 2), ("qkv_gate", 128): (2, 7), ("o", 128): (2, 2),
    ("router", 128): (2, 64),
    ("pooler", 256): (2, 4), ("unpooler", 256): (2, 2), ("qkv_gate", 256): (2, 7), ("o", 256): (2, 2),
    ("router", 256): (2, 64),
}
TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
TMA_KS = 4
USE_TMA = True
USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def _tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


TMA_TU = {}  # (layer, mtk) -> 0 / 1 (which translation unit instantiates it)


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def _tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
    return items


def _tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in _tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], _mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else w_rows[layer]
        wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            TMA_TU[cfg] = 0
    return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


DIAG_LAYERS = ()
GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


def swizzle_tiles(w, bm):
    """W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rb, kb, r, c, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def _gemm_src(sm90):
    if not sm90:
        return ('void gemm90(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, '
                'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "gemm90 needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
    return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    // stage the accumulators in this CTA's (now idle) ring, then one bulk shared::cluster copy into rank 0's buffer
    float* mine = reinterpret_cast<float*>(gbase);   // [NWG][NACC][128]
    if constexpr (NWG > 1) named_bar(1, NWG * 128);   // every warpgroup is done reading the ring
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
}
USE_GQ = True

GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction: after every rank's main loop (cluster barrier: every ring is free), each
// rank owns BN/CL tokens; every thread st.async's the other ranks' token blocks of its accumulator fragment into the
// owner's ring space ([src rank][thread][4 JR floats]) and sums the CL-1 incoming copies of its own block.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(((NB + CL - 1) / CL) * 8 * BM * 2 <= NSTAGE * ST, "gqx: the BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, rbar = bars + 16 * NSTAGE;
  // CL > 1: this rank's accumulator fragments [NB][NTH] float4, outside the ring (written right after the main loop)
  float4* frag = reinterpret_cast<float4*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  float* recv = reinterpret_cast<float*>(gbase);   // [CL][NTH][4 JM] (the ring, once every rank is done with it)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(rbar, 1);

    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barrier
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row0 = rb * BM + wg * 64 + (warp & 3) * 16 + g;
  if constexpr (CL > 1) {
    // 1) fragments out right away (own area: no need to wait for the other ranks' main loops), 2) one cluster
    // barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM, stages BF16 [token][64 rows]
    // in its own (free) ring and 4) writes each token's 64 rows as 128 contiguous bytes; 5) exit barrier
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);   // [owned tokens][BM rows]
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      if (r != (int)crank) continue;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        if (j < (r * NB) / CL || j >= ((r + 1) * NB) / CL) continue;
        float4 pv[CL];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(fl + (uint32_t)((j * NTH + tid) * 16)), "r"(r2));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w) : "r"(ra) : "memory");
        }
        float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          v0 += pv[r2].x; v1 += pv[r2].y; v2 += pv[r2].z; v3 += pv[r2].w;
        }
        const int tl = 8 * (j - (r * NB) / CL) + 2 * t4;   // token within the owned slice
        stg[tl * BM + rl] = __float2bfloat16_rn(v0);
        stg[tl * BM + rl + 8] = __float2bfloat16_rn(v2);
        stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v1);
        stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v3);
      }
    }
    named_bar(1, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row0] = __float2bfloat16_rn(acc[4 * j]);
        out[(size_t)t0 * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row0] = __float2bfloat16_rn(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 + (CL > 1 ? (size_t)(BN / 8) * NWG * 128 * 16 : 0);
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
USE_GQX = True
TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


def _gqx_src(sm90):
    if not sm90:
        return ('void gqx(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqx needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gqx_bn(m):
    return 128 if m <= 128 else 256


def _run_gqx(x, layer, m, n, k, state):
    nwg, cl, ns = GQX_CONFIGS[(layer, _gqx_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqx(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out



RGEMV_SRC = r"""
// ============================================================================ router GEMV (m = 1): FP32 weights
// read directly (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every
// weight and x load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum (deterministic).
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}


// ============================================================================ cgemv (m = 1): 12-bit compressed GEMV
// Lossless planes, row-major: L [N][K] low bytes, C [N][K/2] 4-bit codes (sign | exponent offset from base), byte j of
// each 4-byte group g holding codes 8g + j and 8g + j + 4. Exponents outside the 8-value window decode wrong and are
// corrected exactly by a dedicated escape warp from a per-CTA table (entry 0 = count; ((row << 16) | k, delta bits)).
// Every weight and x load is issued before the first FMA; fixed-order reductions (deterministic).
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 GEMV (the unpooler slice at m = 1: W[:1024] bf16 [1024, 4096]): 4 rows per CTA, 512 threads, thread i owns
// k = 8i (one 16 B load per row and one of x), fixed-order reductions.
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH) sgemv_kernel(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    __nv_bfloat16* __restrict__ out) {
  constexpr int PER = K / NTH / 8;
  constexpr int NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  uint4 w[ROWS][PER], xv[PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) w[r][j] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r0 + r) * K + j * (NTH * 8) + tid * 8));
#pragma unroll
  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat162* hx = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
    float xf[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(hx[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&w[r][j]);
      float v = acc[r];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 a = __bfloat1622float2(hw[e]); v = fmaf(a.x, xf[2 * e], v); v = fmaf(a.y, xf[2 * e + 1], v); }
      acc[r] = v;
    }
  }
  __shared__ float red[ROWS][NWARP];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void sgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % 4 == 0 && N <= W.size(0), "sgemv: m = 1, K = 4096");
  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M) {
  TORCH_CHECK(W.size(1) == 8192 && M == 1, "rgemv: K = 8192, m = 1");
  rgemv_kernel<1, 1><<<(int)W.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), (int)M,
      (int)W.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _gq_src(sm90):
    if not sm90:
        return ('void gq(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gq needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def _run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = GQ_CONFIGS[(layer, _gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gq(state["raw"][layer], x, out, m, n, k, _gq_bn(m), nwg, cl, ns)
    return out



WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def _wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def _wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
        hl = layer == "router"
        if not sm90:
            stages = min(stages, 3 if not hl else 2)  # sm_80 check path: 163 KB of shared memory
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


MID_STAGES = 4
MID_DU = 2


def _mid_src(sm90):
    if not sm90:
        return ('void mid(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, '
                'torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "mid needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits) in []:
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch_mid<{nwg}, {bn}, {MID_STAGES}, {MID_DU}, {str(bool(comp)).lower()}, {ot}>(Wp, x, out, ws, "
                         f"counters, poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    bns = sorted({bn for (_, bn) in MID_CONFIGS})
    return (MID_SRC.replace("@@WGMMA_DEFS@@", wgmma_defs(bns))
            .replace("@@MID_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


RK_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
#endif
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""


def _rk_src(sm90):
    if sm90:
        return RK_SRC
    return 'void rk(at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor) { TORCH_CHECK(false, "rk needs sm_90"); }'


def _get_ext():
    """Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time)."""
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90)
                    + _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC + _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
# CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
# row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
# FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
# SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
# raw); a missing header or a failed build just leaves the route out.
CUTLASS_LAYERS = set()   # rq_lt is 2.66 us faster (27.9 vs 30.56 us); removing CUTLASS saves ~80s prepare time
CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
CUTLASS_SRC = r"""
/***************************************************************************************************
 * The GEMM below instantiates NVIDIA CUTLASS 3.x/4.x templates (header-only, compiled from the CUTLASS headers that
 * ship in this image's FlashInfer package; nothing prebuilt is linked). Its type composition follows CUTLASS's Hopper
 * GEMM examples (examples/48_hopper_warp_specialized_gemm, 49_hopper_gemm_with_collective_builder), so it carries
 * CUTLASS's license notice:
 *
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
int64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  return (int64_t)Gemm::get_workspace_size(args);
}
void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  Gemm gemm;
  TORCH_CHECK((size_t)ws.numel() >= Gemm::get_workspace_size(args), "cut_gemm: workspace");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cut_gemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), st) == cutlass::Status::kSuccess, "cut_gemm: init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "cut_gemm: run");
}
"""
_cut_ext = None
_cut_err = None


def _build_cutlass():
    """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
    global _cut_ext, _cut_err
    try:
        import importlib.util as _ilu
        import os as _os
        major, minor = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError("needs sm_90")
        fi = _ilu.find_spec("flashinfer")
        if fi is None or not fi.submodule_search_locations:
            raise RuntimeError("flashinfer (CUTLASS headers) not installed")
        root = _os.path.join(list(fi.submodule_search_locations)[0], "data", "cutlass")
        inc = [_os.path.join(root, "include"), _os.path.join(root, "tools", "util", "include")]
        if not all(_os.path.isdir(d) for d in inc):
            raise RuntimeError("CUTLASS headers missing under " + root)
        _cut_ext = load_inline(
            name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
            cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                        "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
            cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
            verbose=False)
    except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
        _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


def pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
    # okc_tune (14173ac4) retimed winners, Oct 25 12:30 (search-harness ratios vs production in brackets); pair.py both
    # orders: pooler16 -0.43, unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us (pooler32 r160 / o32 r88: no gain)
    ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
    ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
    ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
    ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
]
SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o16t_r68',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler16t_r192',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8t_r144',
    ('qkv_gate', 16): 'qkv16t_r200',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16t_r72',
    ('unpooler', 32): 'unpooler32_r144',
}
SK2_ROT = False
SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
USE_SK2 = True
WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def _sk2_name(layer, m):
    return SK2_PICK[(layer, _sk2_tokr(m))]


def _sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def _sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def _sk2_src(sm90):
    if not sm90:
        return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }')
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


def sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "wbits": wbits}


CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


def cg_compress(w, rows_per_cta):
    """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)          # row-major order
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}


def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
SK2_BALANCE_LAYERS = {"qkv_gate"}   # v136: only where it measured a gain (qkv m=8/16 -0.5 us; o/unpooler m=32 lost)


def sk2_balance_perm(w, n_rows, R):
    """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    cnt = ((e71 < base) | (e71 > base + 7)).sum(1)
    # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4                                   # groups per block
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def sk2_table_add_perm(tab, pmax, perm, R):
    """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2


def sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax


def sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            perm = None
            if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                pkey = (layer, n_rows, R)
                bal = state.setdefault("sk2_bal", {})
                if pkey not in bal:
                    pr = sk2_balance_perm(w, n_rows, R)
                    bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                perm, comp = bal[pkey]
                if comp is None:
                    perm = None
            if perm is None:
                comp = state.setdefault("sk2_comp", {}).get(layer)
                if comp is None:
                    comp = sk2_compress(w.contiguous())
                    state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                tab, pmax = sk2_table(comp, n_rows, R)
                if perm is not None:
                    tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)
    state.pop("sk2_bal", None)


def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def _mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def prepare(weights):
    import threading
    ext = _get_ext()
    # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
    # after the main build so the two load_inline calls never overlap
    cut_thread = None
    if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
        cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
        cut_thread.start()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
            state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            if not sm90_:   # sm_90 runs the router's wide path on router_cat (TMA tiles); the fragment stream is sm_80's
                state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            del t
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            if name in CG_CONFIGS:
                wc = w[:1024] if name == "unpooler" else w   # the unpooler at m = 1 is the 1024-row slice
                cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                if cg is not None:
                    state.setdefault("cg", {})[name] = cg
            if name in TORCHT_LAYERS and sm90_:
                state.setdefault("rawT", {})[name] = w.t().contiguous()
            c = pack_compressed(w) if name in COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
    state["old_paths"] = old_paths
    sk2_prepare(weights, state)
    if GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in DIAG_LAYERS:
            state["wtile"][layer] = swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({GEMM_CONFIGS[(layer, 128)][1], GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(w_rows[layer] // 16, dtype=torch.int32, device=dev)
    if cut_thread is not None:
        cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
        if cut_thread.is_alive():
            print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
    if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
        state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)   # persistent scheduler: a few KB at most
        state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
    elif _cut_err is not None:
        print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
    _selftest(weights, state)
    rq_prepare(weights, state)
    _autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def _route(layer, m, sm90):
    for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


def _route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
    if route == "skinny":
        return (layer, route, _mt_key(m))
    if route == "wide":
        return (layer, route) + _wide_key(m)
    if route == "rk":
        return (layer, route, (m + 63) // 64)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def _selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in _candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else _route_key(layer, m, ch[0]))
                try:
                    out = _run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = _mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = _get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, w_rows[layer] // 16, ks, dg)
    return out


def _run_wide(x, layer, m, n, k, state):
    tok, nz = _wide_key(m)
    nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
    _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def _run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def _run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def _run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind in ("rq_lt", "rq_tma") or kind.startswith("v228_h_"):
        return rq_run(choice, x, layer, m, n, k, state)
    if kind == "skinny":
        return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return _run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return _run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return _run_gq(x, layer, m, n, k, state)
    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        cg = state["cg"][layer]
        rows, nth = CG_CONFIGS[layer]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
        return out
    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().sgemv(state["raw"][layer], x, out, n)
        return out
    if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rgemv(state["router_f32"], x, out, m)
        return out
    if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
        return out
    if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
        return out
    if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
        return torch.mm(x, state["rawT"][layer])
    return _run_torch(x, layer, m, n, k, state)


RK_SPLITS = {1: 16, 2: 12, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


def _rk_splits(m, nsm=132):
    """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
    ntt = (m + 63) // 64
    return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


def _static_choice(layer, m, state):
    route = _route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                              and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not WIDE_BF16:
            route = "torch"
    if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = _mt_key(m)
        return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def _candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [_static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and _route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
            and _route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if layer == "router" and m == 1 and "router_f32" in state:
        c.insert(0, ("rgemv",))
    if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
            and _route_key(layer, m, "rk") not in state["disabled"]):
        c.insert(0, ("rk",))
    if layer == "unpooler" and m == 1:
        c.insert(0, ("sgemv",))
    if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
        c.insert(0, ("cgemv",))
    if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
        c.append(("torchT",))
    if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
            and _route_key(layer, m, "cutlass") not in state["disabled"]):
        c.append(("cutlass",))
    c.extend(rq_candidates(layer, m, state))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


AUTOTUNE = True
TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def _autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   # 256 MB, read-only sweep
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = _candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                _run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    _run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("v228_h_"):
                incumbents = [i for i, ch in enumerate(cands)
                              if not ch[0].startswith("v228_h_")]
                incumbent = min(incumbents, key=lambda i: med[i])
                candidate_core = max(1000.0 * med[best] - 4.0, 1e-6)
                incumbent_core = max(1000.0 * med[incumbent] - 4.0, 1e-6)
                ratio = candidate_core / incumbent_core
                print("v228 Pooler256 LT gate", cands[best][0],
                      "candidate", 1000.0 * med[best],
                      "incumbent", 1000.0 * med[incumbent],
                      "core_ratio", ratio, flush=True)
                if ratio > 0.99:
                    best = incumbent
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
            if alt:
                state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()
    # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
    for layer, m, tl in state.get("tune_log", []):
        ch = state["choice"][(layer, m)]
        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
        print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
    print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def _nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    if tm == 256 and m != 256 and layer != "router":
        # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
        # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
        # torchT m=240 +11%)
        return ("torch",)
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "cutlass" and m != tm:
        ch = state.get("choice_alt", {}).get((layer, tm))
        if ch is None:
            return None
    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
    out = _run(choice, x, layer, m, n, k, state)
    if GEMM_DIAG:
        _diag(x, layer, m, n, k, state, choice)
    return out


def _diag(x, layer, m, n, k, state, choice):
    """Extra kernels for the judge's per-kernel profile (outputs discarded)."""
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   # same kernel without epilogue / x loads / compute
            _run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


# ----------------------------------------------------------------------------------------------------------------
# Team rohanarun additions (our own work): extra tuner candidates for the m >= 120 shapes where our kernels measured
# faster on H200 (pooler m=128 TMA kernel; cuBLASLt plans with tuned tile/schedule for pooler/qkv/o/unpooler).
# They only enter vjeux's per-shape autotuner as candidates; the tuner keeps whichever is fastest on the judge GPU.
import ctypes as _ct
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def rq_deep_tma_pooler(DX, DW, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     EVEN: tl.constexpr = False):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(K // BK):
        a = DX.load([pm * BM, kb * BK])
        b = DW.load([kb * BK, pn * BN])
        acc = tl.dot(a, b, acc)
    mm = pm * BM + tl.arange(0, BM)
    nn = pn * BN + tl.arange(0, BN)
    if EVEN:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16),
                 (mm[:, None] < M) & (nn[None, :] < N))


class rq_LtAlgo(_ct.Structure):
    _fields_ = [("data", _ct.c_uint64 * 8)]

class rq_LtResult(_ct.Structure):
    _fields_ = [("algo", rq_LtAlgo), ("workspace", _ct.c_size_t),
                ("state", _ct.c_int), ("waves", _ct.c_float),
                ("reserved", _ct.c_int * 4)]

class rq_LtAPI:
    def __init__(self):
        # Let the dynamic loader resolve the installed CUDA library. Avoid
        # ctypes.util.find_library, whose Linux implementation can start processes.
        self.lib = None
        for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
            try:
                self.lib = _ct.CDLL(name)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError("cuBLASLt unavailable")
        V, I, S = _ct.c_void_p, _ct.c_int, _ct.c_size_t
        signatures = {
            "cublasLtCreate": [_ct.POINTER(V)],
            "cublasLtDestroy": [V],
            "cublasLtMatmulDescCreate": [_ct.POINTER(V), I, I],
            "cublasLtMatmulDescSetAttribute": [V, I, V, S],
            "cublasLtMatrixLayoutCreate": [_ct.POINTER(V), I, _ct.c_uint64, _ct.c_uint64, _ct.c_int64],
            "cublasLtMatmulPreferenceCreate": [_ct.POINTER(V)],
            "cublasLtMatmulPreferenceSetAttribute": [V, I, V, S],
            "cublasLtMatmulAlgoGetHeuristic": [V, V, V, V, V, V, V, I, _ct.POINTER(rq_LtResult), _ct.POINTER(I)],
            "cublasLtMatmulAlgoConfigGetAttribute": [_ct.POINTER(rq_LtAlgo), I, V, S, _ct.POINTER(S)],
            "cublasLtMatmulAlgoConfigSetAttribute": [_ct.POINTER(rq_LtAlgo), I, V, S],
            "cublasLtMatmulAlgoCheck": [V, V, V, V, V, V, _ct.POINTER(rq_LtAlgo), _ct.POINTER(rq_LtResult)],
            "cublasLtMatmul": [V, V, V, V, V, V, V, V, V, V, V, V, _ct.POINTER(rq_LtAlgo), V, S, V],
            "cublasLtMatmulPreferenceDestroy": [V],
            "cublasLtMatmulDescDestroy": [V],
            "cublasLtMatrixLayoutDestroy": [V],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = I
        self.handle = V()
        self.check(self.lib.cublasLtCreate(_ct.byref(self.handle)))
        self.alpha, self.beta = _ct.c_float(1.0), _ct.c_float(0.0)

    def __del__(self):
        try:
            handle = getattr(self, "handle", None)
            if handle is not None and handle.value:
                self.lib.cublasLtDestroy(handle)
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError("cuBLASLt status " + str(status))

class rq_LtPlan:
    def __init__(self, api, m, n, k, transposed, tile, custom, pick=False, heuristic_index=None):
        self.api = api
        self.transposed = transposed
        V, I = _ct.c_void_p, _ct.c_int
        self.desc, self.a, self.b, self.c, pref = V(), V(), V(), V(), V()
        lib, check = api.lib, api.check
        check(lib.cublasLtMatmulDescCreate(_ct.byref(self.desc), 68, 0))
        trans = I(0 if transposed else 1)
        check(lib.cublasLtMatmulDescSetAttribute(self.desc, 3, _ct.byref(trans), _ct.sizeof(trans)))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.a), 14, n if transposed else k,
                                             k if transposed else n, n if transposed else k))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.b), 14, k, m, k))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.c), 14, n, m, n))
        check(lib.cublasLtMatmulPreferenceCreate(_ct.byref(pref)))
        # Only select algorithms with no scratch workspace and FP32 reduction.
        workspace = _ct.c_uint64(0)
        reduction = I(2)
        try:
            check(lib.cublasLtMatmulPreferenceSetAttribute(pref, 1, _ct.byref(workspace), _ct.sizeof(workspace)))
            check(lib.cublasLtMatmulPreferenceSetAttribute(pref, 3, _ct.byref(reduction), _ct.sizeof(reduction)))
            result, count = rq_LtResult(), I()
            if heuristic_index is not None:
                many = (rq_LtResult * 48)()
                check(lib.cublasLtMatmulAlgoGetHeuristic(
                    api.handle, self.desc, self.a, self.b, self.c, self.c,
                    pref, 48, many, _ct.byref(count)))
                viable = [i for i in range(count.value)
                          if not many[i].state and not many[i].workspace]
                if heuristic_index >= len(viable):
                    raise RuntimeError("No zero-workspace heuristic at index %d" % heuristic_index)
                selected = many[viable[heuristic_index]].algo
                checked = rq_LtResult()
                check(lib.cublasLtMatmulAlgoCheck(
                    api.handle, self.desc, self.a, self.b, self.c, self.c,
                    _ct.byref(selected), _ct.byref(checked)))
                check(checked.state)
                if checked.workspace:
                    raise RuntimeError("Heuristic unexpectedly needs workspace")
                self.algo = selected
                return
            if pick:
                # Keep a heuristic candidate exactly as returned, selected by its tile and option.
                many = (rq_LtResult * 48)()
                check(lib.cublasLtMatmulAlgoGetHeuristic(api.handle, self.desc, self.a, self.b,
                    self.c, self.c, pref, 48, many, _ct.byref(count)))
                for i in range(count.value):
                    got = []
                    for attribute in (0, 1, 5, 2):
                        val, size = I(), _ct.c_size_t()
                        check(lib.cublasLtMatmulAlgoConfigGetAttribute(_ct.byref(many[i].algo), attribute,
                            _ct.byref(val), _ct.sizeof(val), _ct.byref(size)))
                        got.append(val.value)
                    if not many[i].state and not many[i].workspace and got == [66, tile, custom, 1]:
                        self.algo = many[i].algo
                        return
                raise RuntimeError("No compatible library algorithm")
            check(lib.cublasLtMatmulAlgoGetHeuristic(api.handle, self.desc, self.a, self.b,
                self.c, self.c, pref, 1, _ct.byref(result), _ct.byref(count)))
            if count.value != 1 or result.state:
                raise RuntimeError("No compatible library algorithm")
            algorithm_id, size = I(), _ct.c_size_t()
            check(lib.cublasLtMatmulAlgoConfigGetAttribute(_ct.byref(result.algo), 0,
                _ct.byref(algorithm_id), _ct.sizeof(algorithm_id), _ct.byref(size)))
            if algorithm_id.value != 66:
                raise RuntimeError("Unsupported library algorithm")
            for attribute, value in ((1, tile), (6, 35), (2, 1), (3, 0), (5, custom)):
                v = I(value)
                check(lib.cublasLtMatmulAlgoConfigSetAttribute(_ct.byref(result.algo),
                    attribute, _ct.byref(v), _ct.sizeof(v)))
            checked = rq_LtResult()
            check(lib.cublasLtMatmulAlgoCheck(api.handle, self.desc, self.a, self.b,
                self.c, self.c, _ct.byref(result.algo), _ct.byref(checked)))
            check(checked.state)
            if checked.workspace:
                raise RuntimeError("Unexpected workspace requirement")
            self.algo = result.algo
        finally:
            lib.cublasLtMatmulPreferenceDestroy(pref)

    def __del__(self):
        try:
            self.api.lib.cublasLtMatmulDescDestroy(self.desc)
            for layout in (self.a, self.b, self.c):
                self.api.lib.cublasLtMatrixLayoutDestroy(layout)
        except (AttributeError, TypeError):
            pass

    def run(self, x, weight, out):
        api, V = self.api, _ct.c_void_p
        # These pointers are constructed from this invocation, never stored in a plan.
        status = api.lib.cublasLtMatmul(api.handle, self.desc, _ct.byref(api.alpha),
            V(weight.data_ptr()), self.a, V(x.data_ptr()), self.b, _ct.byref(api.beta),
            V(out.data_ptr()), self.c, V(out.data_ptr()), self.c, _ct.byref(self.algo),
            None, 0, V(torch.cuda.current_stream(x.device).cuda_stream))
        if status in (7, 15):
            return False
        api.check(status)
        return True


RQ_LT = [   # (layer, m, weight pre-transposed, tile id, custom option, pick from heuristic list)
    ("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 134, 2, True), ("o", 256, False, 20, 4, False),
    ("o", 128, False, 18, 2, False), ("unpooler", 128, True, 18, 4, False), ("unpooler", 256, True, 20, 4, False),
]
RQ_TMA = {("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}
_V228_ROUTES = (("pooler", 256),)
_V228_HEURISTICS = 12


def rq_prepare(weights, state):
    rq = {"t": {}, "plans": {}, "v228_heuristic_plans": {}}
    state["rq"] = rq
    try:
        for name in ("pooler", "unpooler", "o", "qkv_gate"):
            rq["t"][name] = weights[name].t().contiguous()
        api = rq_LtAPI()
        rq["api"] = api
        for layer, m, tr, tile, custom, pick in RQ_LT:
            n, k = weights[layer].shape
            try:
                rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)
            except Exception:
                pass
        for layer, m in _V228_ROUTES:
            n, k = weights[layer].shape
            for transposed in (False, True):
                for index in range(_V228_HEURISTICS):
                    name = "v228_h_%s_%d_%s_%02d" % (
                        layer, m, "t" if transposed else "n", index)
                    try:
                        rq["v228_heuristic_plans"][name] = rq_LtPlan(
                            api, m, n, k, transposed, 0, 0,
                            heuristic_index=index)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[okc] rq_prepare: {type(e).__name__}: {e}", file=sys.stderr)


def rq_candidates(layer, m, state):
    rq = state.get("rq", {})
    c = []
    if (layer, m) in RQ_TMA and layer in rq.get("t", {}):
        c.append(("rq_tma",))
    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    prefix = "v228_h_%s_%d_" % (layer, m)
    c.extend((name,) for name in sorted(rq.get("v228_heuristic_plans", {}))
             if name.startswith(prefix))
    return c


def rq_run(choice, x, layer, m, n, k, state):
    rq = state["rq"]
    if choice[0] == "rq_lt" or choice[0].startswith("v228_h_"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("v228_heuristic_plans", {}).get(choice[0]))
        if plan is not None and x.is_contiguous():
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            w = rq["t"][layer] if plan.transposed else state["raw"].get(layer, None)
            if w is None:
                w = rq["t"][layer].t()
            if plan.run(x, w, out):
                return out
        return _run_torch(x, layer, m, n, k, state)
    # rq_tma: TMA + tl.dot pooler/unpooler kernel on the K-major weight copy (any m)
    bm, bn, bk, stg = RQ_TMA.get((layer, m), RQ_TMA.get((layer, 256 if m > 128 else 128)))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    dx = TensorDescriptor.from_tensor(x, [bm, bk])
    dw = TensorDescriptor.from_tensor(rq["t"][layer], [bk, bn])
    rq_deep_tma_pooler[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dx, dw, out, m, n, k, bm, bn, bk, EVEN=(m % bm == 0 and n % bn == 0), num_warps=4, num_stages=stg)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Per-shape composition: the shapes in _ALT_ROUTES run the embedded readable alternate source module (its own namespace, its own
# prepare state); every other shape, and every m outside the scored set, runs the code above.
# Portions of this file (the embedded readable alternate source module) are adapted from software under the MIT License:
#   Copyright (c) 2026 Nathen Fernandes
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
#   notice and this permission notice shall be included in all copies or substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import types as _types
_ALT_ROUTES = frozenset([('o', 8), ('o', 128), ('o', 256), ('pooler', 8), ('pooler', 16), ('qkv_gate', 128), ('router', 8), ('router', 16), ('router', 32), ('unpooler', 8), ('unpooler', 16)])

def _v179_build_alt_namespace():
    from typing import TypedDict
    import hashlib
    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load_inline

    class Weights(TypedDict):
        qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
        o: torch.Tensor         # bfloat16 [8192, 1024]
        pooler: torch.Tensor    # bfloat16 [4096, 8192]
        unpooler: torch.Tensor  # bfloat16 [8192, 4096]
        router: torch.Tensor    # float32 [256, 8192]

    def _mma(n):
        regs = ', '.join(f'%{i}' for i in range(n // 2))
        outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
        return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
    }}'''

    ROUTER_CUDA = (
        '\n'
        '#include <cuda.h>\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#define RD __device__ __forceinline__\n'
        'RD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        'RD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }\n'
        'RD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }\n'
        'RD void rr_wait(uint32_t b, int ph) {\n'
        '  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");\n'
        '}\n'
        'RD void rr_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }\n'
        'RD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }\n'
        'RD void rr_push(uint32_t d, float4 v, uint32_t b) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n'
        '}\n'
        'RD float4 rr_pull(uint32_t d) {\n'
        '  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;\n'
        '}\n'
        'RD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }\n'
        'RD uint64_t rr_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }\n'
        'RD void rr_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {\n'
        '  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");\n'
        '}\n'
        '@@MMA@@\n'
        '\n'
        'template<int BN,int S,int ST,int SW,int RED,int PIPE>\n'
        '__global__ __launch_bounds__((PIPE&2)?256:128) void rr_kernel(const __grid_constant__ CUtensorMap tw,\n'
        '                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {\n'
        '  constexpr int NR=SW?64:BN, TM=SW?BN:64;\n'
        '  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;\n'
        '  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;\n'
        '  extern __shared__ __align__(1024) unsigned char sm[];\n'
        '  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;\n'
        '  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;\n'
        '  const uint32_t base=rr_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;\n'
        '  const uint32_t recv=rb+16;\n'
        '  const int kb=rank*(128/S);\n'
        '  auto load=[&](int i) {\n'
        '    const int st=i%ST;\n'
        '    const uint32_t b=bars+st*8, d=base+st*SB;\n'
        '    rr_expect(b,SB);\n'
        '    rr_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);\n'
        '    rr_tma(d+2*WB,&tx,(kb+i)*64,tok,b);\n'
        '  };\n'
        '  if(tid==((PIPE&2)?128:0)) {\n'
        '#pragma unroll\n'
        '    for(int i=0;i<ST;i++) { rr_init(bars+i*8,1); if constexpr(PIPE&2) rr_init(empty+i*8,4); }\n'
        '    asm volatile("fence.proxy.async.shared::cta;" :::"memory");\n'
        '    if constexpr(PIPE&4) {\n'
        '    if constexpr(RED>=1) {\n'
        '      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);\n'
        '      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");\n'
        '    }\n'
        '    }\n'
        '#pragma unroll\n'
        '    for(int i=0;i<ST && i<128/S;i++) load(i);\n'
        '    if constexpr(!(PIPE&4)) {\n'
        '    if constexpr(RED>=1) {\n'
        '      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);\n'
        '      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");\n'
        '    }\n'
        '    }\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");\n'
        '  if constexpr(PIPE&2) {\n'
        '    if(tid>=128) {\n'
        '      if(tid==128) {\n'
        '        for(int i=ST;i<128/S;i++) {\n'
        '          rr_wait(empty+(i%ST)*8,((i/ST)-1)&1);\n'
        '          load(i);\n'
        '        }\n'
        '      }\n'
        '      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n'
        '      if constexpr(RED!=2) rr_sync();\n'
        '      return;\n'
        '    }\n'
        '  }\n'
        '  float acc[BN/2];\n'
        '#pragma unroll\n'
        '  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));\n'
        '  int prev=0;\n'
        '#pragma unroll 1\n'
        '  for(int i=0;i<128/S;i++) {\n'
        '    const int st=i%ST;\n'
        '    rr_wait(bars+st*8,(i/ST)&1);\n'
        '    const uint32_t w=base+st*SB, x=w+2*WB;\n'
        '#pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");\n'
        '    asm volatile("wgmma.fence.sync.aligned;" :::"memory");\n'
        '#pragma unroll\n'
        '    for(int q=0;q<4;q++) {\n'
        '      const uint64_t dx=rr_desc(x+q*32),dh=rr_desc(w+q*32),dl=rr_desc(w+WB+q*32);\n'
        '      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }\n'
        '      else { @@CALL_NH@@; @@CALL_NL@@; }\n'
        '    }\n'
        '    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");\n'
        '    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");\n'
        '    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");\n'
        '#pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");\n'
        '    if constexpr(PIPE&2) {\n'
        '      if constexpr(PIPE&1) {\n'
        '        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");\n'
        '      } else {\n'
        '        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");\n'
        '      }\n'
        '    } else if constexpr(PIPE&1) {\n'
        '      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);\n'
        '    } else {\n'
        '      if(i+ST<128/S && tid==0) load(i+ST);\n'
        '    }\n'
        '    prev=i;\n'
        '  }\n'
        '  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");\n'
        '  auto emit=[&](int v,float4 z) {\n'
        '    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;\n'
        '    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;\n'
        '    const float vv[4]={z.x,z.y,z.z,z.w};\n'
        '#pragma unroll\n'
        '    for(int c=0;c<4;c++) {\n'
        '      const int a=ar+(c/2)*8, b=bc+(c&1);\n'
        '      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);\n'
        '      if(tr<m) out[tr*256+wc]=vv[c];\n'
        '    }\n'
        '  };\n'
        '  if constexpr(S==1) {\n'
        '#pragma unroll\n'
        '    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));\n'
        '  } else if constexpr(RED==0) {\n'
        '    float4* part=(float4*)sm;\n'
        '#pragma unroll\n'
        '    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);\n'
        '    rr_sync();\n'
        '    for(int t=tid;t<OWN;t+=128) {\n'
        '      const int v=rank*OWN+t;\n'
        '      float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '      for(int r=0;r<S;r++) rr_add(sum,rr_pull(rr_map(base+v*16,r)));\n'
        '      emit(v,sum);\n'
        '    }\n'
        '    rr_sync();\n'
        '  } else {\n'
        '    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n'
        '#pragma unroll\n'
        '    for(int j=0;j<BN/8;j++) {\n'
        '      const int v=j*128+tid, dst=v/OWN, local=v%OWN;\n'
        '      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);\n'
        '      const uint32_t addr=recv+(rank*OWN+local)*16;\n'
        '      if(dst==rank) *(float4*)(sm+(addr-base))=val;\n'
        '      else rr_push(rr_map(addr,dst),val,rr_map(rb,dst));\n'
        '    }\n'
        '    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();\n'
        '    rr_wait(rb,0);\n'
        '    for(int t=tid;t<OWN;t+=128) {\n'
        '      float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '      for(int r=0;r<S;r++) rr_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));\n'
        '      emit(rank*OWN+t,sum);\n'
        '    }\n'
        '    if constexpr(RED!=2) rr_sync();\n'
        '  }\n'
        '}\n'
        '\n'
        'typedef CUresult (*RR_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,\n'
        ' const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);\n'
        'static int rr_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box) {\n'
        '  static RR_Encode enc=nullptr;\n'
        '  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }\n'
        '  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};\n'
        '  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};\n'
        '  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n'
        '                 CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n'
        '}\n'
        '\n'
        'template<int BN,int S,int ST,int SW,int RED,int PIPE>\n'
        'int rr_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  constexpr int NR=SW?64:BN, TM=SW?BN:64;\n'
        '  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);\n'
        '  auto fn=rr_kernel<BN,S,ST,SW,RED,PIPE>;\n'
        '  static bool configured=false;\n'
        '  if(!configured) {\n'
        '    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);\n'
        '    if(e!=cudaSuccess) return -(int)e;\n'
        '    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);\n'
        '    if(e!=cudaSuccess)return -(int)e;\n'
        '    configured=true;\n'
        '  }\n'
        '  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n'
        '  cudaLaunchConfig_t cfg={};\n'
        '  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n'
        '  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};\n'
        '  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}\n'
        '  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}\n'
        '  CUtensorMap tw,tx;\n'
        '  int e=rr_map_host(&tw,w,64,256*128*2,NR*2);if(e)return e;\n'
        '  e=rr_map_host(&tx,x,8192,m,TM);if(e)return e;\n'
        '  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);\n'
        '  return err==cudaSuccess?0:(int)err;\n'
        '}\n'
        'int rr_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        ' switch(id){@@CASES@@ default:return -99;}\n'
        '}\n'
        '@@SCALAR@@\n'
    )

    ROUTER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);\nint rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

    def build_router(configs):
        src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
        for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
            src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
        cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
        src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
        name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
        return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                           extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                           no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)

    def pack_router(w, nr):
        hi = w.to(torch.bfloat16)
        lo = (w-hi.float()).to(torch.bfloat16)
        return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)

    def prepare_candidates(weights, configs):
        w = weights['router'].contiguous()
        return {'ext':build_router(configs), 'configs':configs,'weight':w,
                'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}

    def run_candidate(x, st, index):
        out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
        stream = torch.cuda.current_stream().cuda_stream
        if index == -1:
            rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
        else:
            cfg = st['configs'][index]
            w = st['packed'][64 if cfg[3] else cfg[0]]
            rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
        if rc:
            raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
        return out

    ROUTER_CONFIGS = [(32, 16, 4, 1, 2, 1), (32, 8, 4, 0, 2, 1), (64, 4, 8, 1, 2, 3)]

    def prepare_router(weights) -> dict:
        w = weights.get('router')
        if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
            return {'enabled':False}
        if torch.cuda.get_device_capability(w.device) != (9,0):
            return {'enabled':False}
        st = prepare_candidates(weights, ROUTER_CONFIGS)
        st['small'] = {(16,8):pack_small_router(w,16,8)}
        st['enabled'] = True
        for m in (1,8,16,32,128,256):
            x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
            run_router(x,'router',256,st)
        torch.cuda.synchronize(w.device)
        return st

    def run_router(x, layer, n, st) -> torch.Tensor | None:
        if not st.get('enabled') or layer != 'router' or n != 256:
            return None
        if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
            return None
        if x.device != st['weight'].device or x.data_ptr() % 16:
            return None
        m = x.shape[0]
        if not 1 <= m <= 256:
            return None
        if m == 1:
            return run_candidate(x,st,-1)
        if m <= 16:
            return run_small_candidate(x,st)
        return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)

    ROUTER_SCALAR = (
        '\n'
        'template<int M, int NT, int ST>\n'
        '__global__ void rr_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n'
        '                                float* __restrict__ y, int m) {\n'
        '  constexpr int K=8192;\n'
        '  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n'
        '  float acc[M][4]={};\n'
        '  float4 ww[ST];\n'
        '  uint2 xx[ST][M];\n'
        '  auto ld=[&](int slot,int i) {\n'
        '    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) {\n'
        '      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n'
        '    }\n'
        '  };\n'
        '  #pragma unroll\n'
        '  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n'
        '  #pragma unroll 1\n'
        '  for(int base=tid;base<K/4;base+=ST*NT) {\n'
        '    #pragma unroll\n'
        '    for(int s=0;s<ST;s++) {\n'
        '      float4 v=ww[s];\n'
        '      uint2 a[M];\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) a[t]=xx[s][t];\n'
        '      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) if(t<m) {\n'
        '        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n'
        '        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n'
        '        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n'
        '        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n'
        '        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n'
        '        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n'
        '        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '  __shared__ float sums[M][NT/32];\n'
        '  #pragma unroll\n'
        '  for(int t=0;t<M;t++) if(t<m) {\n'
        '    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n'
        '    #pragma unroll\n'
        '    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '    if(lane==0) sums[t][warp]=v;\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if(warp==0) {\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) if(t<m) {\n'
        '      float v=lane<NT/32?sums[t][lane]:0.f;\n'
        '      #pragma unroll\n'
        '      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '      if(lane==0) y[t*256+row]=v;\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'int rr_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  auto kern=rr_scalar_kernel<1,512,2>;\n'
        '  if(query==2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n'
        '    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n'
        '  }\n'
        '  if(m!=1) return -5;\n'
        '  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n'
        '  return cudaGetLastError()==cudaSuccess?0:-8;\n'
        '}\n'
    )

    ROUTER_SMALL = (
        '\n'
        'template<int MT,int S,int WK>\n'
        '__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {\n'
        '  constexpr int KU=128/S/WK, NI=MT*32;\n'
        '  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;\n'
        '  const int gw=(rank*16+row/16)*WK+warp;\n'
        '  extern __shared__ __align__(16) unsigned char sm[];\n'
        '  float4* recv=(float4*)sm;\n'
        '  float4* part=recv+S*NI;\n'
        '  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);\n'
        '  int4 w[KU][8];uint4 xx[KU][MT][2];\n'
        '#pragma unroll\n'
        '  for(int k=0;k<KU;k++) {\n'
        '#pragma unroll\n'
        '    for(int j=0;j<8;j++) {\n'
        '      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;\n'
        '      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));\n'
        '    }\n'
        '#pragma unroll\n'
        '    for(int n=0;n<MT;n++) {\n'
        '      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;\n'
        '#pragma unroll\n'
        '      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));\n'
        '    }\n'
        '  }\n'
        '  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}\n'
        '  if(rank==0 && warp==0) __syncwarp();\n'
        '  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}\n'
        '  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");\n'
        '  float c[2][MT][4]={};\n'
        '#pragma unroll\n'
        '  for(int k=0;k<KU;k++) {\n'
        '#pragma unroll\n'
        '    for(int n=0;n<MT;n++) {\n'
        '      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};\n'
        '#pragma unroll\n'
        '      for(int q=0;q<4;q++) {\n'
        '#pragma unroll\n'
        '        for(int h=0;h<2;h++) {\n'
        '          const int4 a=w[k][q+h*4];\n'
        '          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '#pragma unroll\n'
        '  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);\n'
        '  __syncthreads();\n'
        '  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n'
        '  for(int v=tid;v<NI;v+=WK*32) {\n'
        '    float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);\n'
        '    if(rank==0)recv[v]=sum;\n'
        '    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));\n'
        '  }\n'
        '  if(rank==0) {\n'
        '    __syncthreads();rr_wait(rb,0);\n'
        '    for(int v=tid;v<NI;v+=WK*32) {\n'
        '      float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);\n'
        '      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);\n'
        '      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}\n'
        '      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'template<int MT,int S,int WK>\n'
        'int rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  auto fn=rr_small_kernel<MT,S,WK>;\n'
        '  constexpr int bytes=(S+WK)*MT*512+16;\n'
        '  static bool configured=false;\n'
        '  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}\n'
        '  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n'
        '  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n'
        '  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;\n'
        '  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);\n'
        '}\n'
        'int rr_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {\n'
        '#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return rr_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rr_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rr_small_t<4,S,W>(w,x,out,m,stream,query);}\n'
        'RSM(8,8) RSM(16,8) RSM(16,4)\n'
        'return -99;\n'
        '}\n'
        '\n'
    )

    def pack_small_router(w, s=16, wk=8):
        hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
        def units(v):
            v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
            return v.permute(0,1,3,2,4)
        data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
        ku=128//s//wk
        return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()

    def run_small_candidate(x,st,s=16,wk=8):
        out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
        rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
        if rc:raise RuntimeError(f'router small launch failed: {rc}')
        return out

    OLAYER_CUDA = (
        '\n'
        '#include <cuda.h>\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#define OD __device__ __forceinline__\n'
        'OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        'OD void ol_wait(uint32_t b,int ph) {\n'
        '  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");\n'
        '}\n'
        'OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\n'
        'OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\n'
        'OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {\n'
        '  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");\n'
        '}\n'
        '@@MMA@@\n'
        '\n'
        'template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n'
        '__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(\n'
        '    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n'
        '    const uint8_t* w,__nv_bfloat16* out,int m) {\n'
        '  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;\n'
        '  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);\n'
        '  extern __shared__ __align__(1024) uint8_t sm[];\n'
        '  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;\n'
        '  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;\n'
        '  uint64_t pw,px;\n'
        '  if constexpr(HINT) {\n'
        '    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));\n'
        '    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));\n'
        '  } else {\n'
        '    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;\n'
        '  }\n'
        '  auto load=[&](int i) {\n'
        '    int s=i%ST;\n'
        '    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n'
        '    if constexpr(PACK==1) {\n'
        '      const uint8_t* src=w+((long)nt*IT+i)*AB;\n'
        '      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");\n'
        '    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);\n'
        '    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);\n'
        '    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n'
        '  };\n'
        '  if(tid==((PIPE&2)?CT:0)) {\n'
        '    #pragma unroll\n'
        '    for(int s=0;s<ST;s++) {\n'
        '      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n'
        '      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");\n'
        '    }\n'
        '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n'
        '    #pragma unroll\n'
        '    for(int i=0;i<ST && i<IT;i++) load(i);\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if((PIPE&2) && tid>=CT) {\n'
        '    if(tid==CT) {\n'
        '      #pragma unroll 1\n'
        '      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n'
        '    }\n'
        '  } else {\n'
        '    float a[BN/2];\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");\n'
        '    #pragma unroll 1\n'
        '    for(int i=0;i<IT;i++) {\n'
        '      int s=i%ST;\n'
        '      ol_wait(full+s*8,(i/ST)&1);\n'
        '      #pragma unroll\n'
        '      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n'
        '      #pragma unroll\n'
        '      for(int kk=0;kk<BK/16;kk++) {\n'
        '        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);\n'
        '        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);\n'
        '        @@CALL@@\n'
        '      }\n'
        '      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n'
        '      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n'
        '      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n'
        '      #pragma unroll\n'
        '      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '      int done=i-((PIPE&1)?1:0);\n'
        '      if(done>=0 && done+ST<IT && tid%128==0) {\n'
        '        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);\n'
        '        else load(done+ST);\n'
        '      }\n'
        '    }\n'
        '    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/8;j++) {\n'
        '      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);\n'
        '      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n'
        '      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);\n'
        '      uint32_t d=base+2*(cb*72+rb);\n'
        '      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");\n'
        '    }\n'
        '  }\n'
        '  __syncthreads();\n'
        '  for(int q=tid;q<T*(R/8);q+=TH) {\n'
        '    int token=q/(R/8),row=(q%(R/8))*8;\n'
        '    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);\n'
        '  }\n'
        '}\n'
        '\n'
        'using OLEncode=decltype(&cuTensorMapEncodeTiled);\n'
        'int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {\n'
        '  static OLEncode encode=nullptr;\n'
        '  if(!encode) {\n'
        '    cudaDriverEntryPointQueryResult q;\n'
        '    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);\n'
        '    if(rc!=cudaSuccess || !encode) return -100;\n'
        '  }\n'
        '  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};\n'
        '  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};\n'
        '  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n'
        '}\n'
        'template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n'
        'int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n'
        '  constexpr int T=BN*TW,R=PACK==2?32:64;\n'
        '  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;\n'
        '  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;\n'
        '  static bool configured=false;\n'
        '  if(!configured) {\n'
        '    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n'
        '    configured=true;\n'
        '  }\n'
        '  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n'
        '  CUtensorMap tw,tx;\n'
        '  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n'
        '  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;\n'
        '  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n'
        '  return int(cudaGetLastError());\n'
        '}\n'
        'int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n'
        '  switch(id) {@@CASES@@ default:return -99;}\n'
        '}\n'
    )

    OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

    def olayer_source(configs) -> str:
        ns = sorted({c[0] for c in configs})
        functions = []
        for n in ns:
            regs = ','.join(f'%{i}' for i in range(n // 2))
            outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
            functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
              asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
            }}''')
        cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
        calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
        return OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)

    def build_olayer(configs):
        for cfg in configs:
            bn,bk,st,pipe,pack,hint,*groups = cfg
            tw = groups[0] if groups else 1
            if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                    or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                    or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                    or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
                raise ValueError(f'unsupported olayer configuration: {cfg}')
        source = olayer_source(configs)
        name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
        return load_inline(name,OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                           extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                           no_implicit_headers=True,with_pytorch_error_handling=False)

    def pack_olayer(weight) -> torch.Tensor:
        columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
        tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
        return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()

    def prepare_olayer_candidates(weights,configs) -> dict:
        weight = weights['o'].contiguous()
        return {'enabled':True,'ext':build_olayer(configs),'configs':configs,'weight':weight,
                'packed':pack_olayer(weight) if any(c[4]==1 for c in configs) else None,
                'tiled32':weight.reshape(256,32,16,64).permute(0,2,1,3).contiguous() if any(c[4]==2 for c in configs) else None}

    def run_olayer_candidate(x,st,index) -> torch.Tensor:
        pack = st['configs'][index][4]
        weight = st['tiled32'] if pack==2 else st['packed'] if pack==1 else st['weight']
        out = torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
        rc = st['ext'].ol_launch(index,x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
        if rc:
            raise RuntimeError(f'olayer launch failed: {rc}, config={st["configs"][index]}')
        return out

    tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = (
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#include <type_traits>\n'
        '#include <cooperative_groups.h>\n'
        '\n'
        '#define SMALLM2D __device__ __forceinline__\n'
        '\n'
        'SMALLM2D int4 smallm2_ldw(const int4* p) {\n'
        '  int4 v;\n'
        '  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n'
        '               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMALLM2D uint4 smallm2_ldx(const void* p) {\n'
        '  uint4 v;\n'
        '  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMALLM2D uint32_t smallm2_lde(const uint32_t* p) {\n'
        '  uint32_t v;\n'
        '  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {\n'
        '  uint32_t d;\n'
        '  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n'
        '  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n'
        '      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n'
        '}\n'
        '\n'
        '// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\n'
        'SMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n'
        '  const uint32_t cm = C & 0x77777777u;\n'
        '  const uint32_t hlo = smallm2_prmt(T0, T1, cm);\n'
        '  const uint32_t hhi = smallm2_prmt(T0, T1, cm >> 16);\n'
        '  a[0] = smallm2_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n'
        '  a[1] = smallm2_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n'
        '  a[2] = smallm2_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n'
        '  a[3] = smallm2_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n'
        '}\n'
        '\n'
        '__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }\n'
        '\n'
        '// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\n'
        'SMALLM2D void smallm2_order_point(int m, void* p) {\n'
        '  uint32_t t;\n'
        '  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n'
        '  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);\n'
        '}\n'
        '\n'
        'SMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        '\n'
        'SMALLM2D uint32_t smallm2_rank() {\n'
        '  uint32_t r;\n'
        '  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n'
        '  return r;\n'
        '}\n'
        '\n'
        'SMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {\n'
        '  uint32_t d;\n'
        '  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {\n'
        '  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {\n'
        '  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\n'
        'SMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\n'
        'SMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n'
        '\n'
        'SMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n'
        '               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n'
        '}\n'
        '\n'
        'SMALLM2D bool smallm2_mbar_try(uint32_t bar) {\n'
        '  uint32_t ok;\n'
        '  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n'
        '               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n'
        '  return ok != 0;\n'
        '}\n'
        '\n'
        'SMALLM2D uint64_t smallm2_gtime() {\n'
        '  uint64_t t;\n'
        '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n'
        '  return t;\n'
        '}\n'
        '\n'
        '// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\n'
        'SMALLM2D void smallm2_wait_bar(uint32_t bar) {\n'
        '  if (smallm2_mbar_try(bar)) return;\n'
        '  const uint64_t t0 = smallm2_gtime();\n'
        '  while (!smallm2_mbar_try(bar))\n'
        '    if (smallm2_gtime() - t0 > 20000000ull) __trap();\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n'
        '\n'
        'SMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {\n'
        '  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n'
        '                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n'
        '}\n'
        '\n'
        'SMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\n'
        'SMALLM2D void smallm2_store(float* p, float v) { *p = v; }\n'
        '\n'
        'template <int FMT> struct Smallm2Fmt;\n'
        'template <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };\n'
        'template <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };\n'
        'template <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };\n'
        'template <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };\n'
        '\n'
        'template <int FMT>\n'
        'SMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n'
        '                   uint32_t U0, uint32_t U1) {\n'
        '  if constexpr (FMT == 0 || FMT == 2) {\n'
        '#pragma unroll\n'
        '    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {\n'
        '      A[4 * l + 0] = (uint32_t)w[l].x;\n'
        '      A[4 * l + 1] = (uint32_t)w[l].y;\n'
        '      A[4 * l + 2] = (uint32_t)w[l].z;\n'
        '      A[4 * l + 3] = (uint32_t)w[l].w;\n'
        '    }\n'
        '  } else {\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {\n'
        '      const int4* u = w + 3 * h;\n'
        '      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n'
        '      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n'
        '      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n'
        '      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n'
        '      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        "// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.\n"
        'template <int NR>\n'
        'SMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {\n'
        '  const uint32_t reg = (e >> 8) & 31u;\n'
        '  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n'
        '#pragma unroll\n'
        '  for (int r = 0; r < NR; ++r)\n'
        '    if (reg == (uint32_t)r) A[r] ^= xr;\n'
        '}\n'
        '\n'
        "// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.\n"
        'template <class Emit>\n'
        'SMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n'
        '  for (int idx = tid; idx < NI; idx += nt) {\n'
        '    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n'
        '    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);\n'
        '    emit(idx, v);\n'
        '  }\n'
        '}\n'
        '\n'
        '// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n'
        '// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n'
        '// straight-line code block at the end, 128 (with 4) no output stores.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n'
        '__global__ void __launch_bounds__(NTMAX) smallm2_gemv(\n'
        '    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n'
        '    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n'
        '    uint32_t U0, uint32_t U1) {\n'
        '  const int m = CM ? CM : m_arg;\n'
        '  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n'
        '  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n'
        '  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;\n'
        '  constexpr bool C12 = FMT == 1 || FMT == 3;\n'
        '  constexpr bool HILO = FMT >= 2;\n'
        '  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n'
        '  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n'
        '  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n'
        '  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n'
        '  OutT* out = reinterpret_cast<OutT*>(outp);\n'
        '  extern __shared__ __align__(16) unsigned char smem[];\n'
        '  __shared__ __align__(8) uint64_t s_bar;\n'
        '  if constexpr (EMPTY) {\n'
        '    return;\n'
        '  } else {\n'
        '    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n'
        '    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n'
        '    const int S = CS ? CS : gridDim.y;\n'
        '    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n'
        '    const int G = GX * S * NW;\n'
        '    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n'
        '    const int rt0 = (blockIdx.x * WR + wr) * RT;\n'
        '    const int c0 = (blockIdx.y * WK + wk) * KU;\n'
        '    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);\n'
        '    const uint32_t bar = smallm2_smem_u32(&s_bar);\n'
        '    uint32_t rank = 0, my_e = 0xFFFFFFFFu;\n'
        '    if constexpr (!(OPT & 1)) {\n'
        '    if constexpr (PUSH) {\n'
        '      rank = smallm2_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        smallm2_mbar_init(bar, 1);\n'
        '        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n'
        '        smallm2_fence_init();\n'
        '      }\n'
        '      smallm2_cl_arrive();\n'
        '    }\n'
        '\n'
        '    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    }\n'
        '\n'
        '    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n'
        '    const size_t sstride = (size_t)G * (RT * NL * 32);\n'
        '    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n'
        '\n'
        '    int4 w[ST][RT][NL];\n'
        '    uint4 xv[ST][MT][2];\n'
        '    auto load = [&](int slot, int s) {\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n'
        '      if constexpr (!NOX && !RO && !(OPT & 2)) {\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n) {\n'
        '          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n'
        '          xv[slot][n][0] = smallm2_ldx(xp);\n'
        '          xv[slot][n][1] = smallm2_ldx(xp + 8);\n'
        '        }\n'
        '      }\n'
        '    };\n'
        '#pragma unroll\n'
        '    for (int s = 0; s < ST; ++s) load(s, s);\n'
        '    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n'
        '    if constexpr (OPT & 1) {\n'
        '    if constexpr (PUSH) {\n'
        '      rank = smallm2_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        smallm2_mbar_init(bar, 1);\n'
        '        smallm2_mbar_expect(bar, parts * RT * TV * 128u);\n'
        '        smallm2_fence_init();\n'
        '      }\n'
        '      smallm2_cl_arrive();\n'
        '    }\n'
        '\n'
        '    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    }\n'
        '\n'
        '    float acc[NA][RT][MT][4];\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < NA; ++h)\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n'
        '    uint32_t fold = 0;\n'
        '    uint32_t nxt = 0xFFFFFFFFu;\n'
        '    int ep = 0;\n'
        '    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n'
        '\n'
        '#pragma unroll 1\n'
        '    for (int i = 0; i < KU; i += ST) {\n'
        '#pragma unroll\n'
        '      for (int s = 0; s < ST; ++s) {\n'
        '        const int si = i + s;\n'
        '        if (ST == 3 && si >= KU) break;\n'
        '        if constexpr (RO) {\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int l = 0; l < NL; ++l)\n'
        '              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n'
        '        } else {\n'
        '          uint32_t b[MT][8];\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n) {\n'
        '            if constexpr (NOX) {\n'
        '#pragma unroll\n'
        '              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n'
        '            } else if constexpr (OPT & 2) {\n'
        '              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;\n'
        '              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);\n'
        '              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;\n'
        '              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;\n'
        '            } else {\n'
        '              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n'
        '              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n'
        '            }\n'
        '          }\n'
        '          uint32_t A[RT][NA * 16];\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);\n'
        '          if constexpr (C12) {\n'
        '#pragma unroll\n'
        '            for (int j = 0; j < RT; ++j) {\n'
        '              const uint32_t useq = (uint32_t)(si * RT + j);\n'
        '              if ((nxt >> 18) == useq) {\n'
        '                do {\n'
        '                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);\n'
        '                  ++ep;\n'
        '                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n'
        '                } while ((nxt >> 18) == useq);\n'
        '              }\n'
        '            }\n'
        '          }\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '              for (int q = 0; q < 4; ++q) {\n'
        '                if constexpr (NOMMA) {\n'
        '#pragma unroll\n'
        '                  for (int h = 0; h < NA; ++h)\n'
        '                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n'
        '                            A[j][16 * h + 4 * q + 3];\n'
        '                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n'
        '                } else {\n'
        '                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                }\n'
        '              }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '\n'
        '    if constexpr (RO) {\n'
        '      if constexpr (RO_CODE) {\n'
        '        if (warp == 0) {\n'
        '          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n'
        '          const uint32_t k = (uint32_t)m | 1u;\n'
        '#pragma unroll\n'
        '          for (int q = 0; q < 64; ++q)\n'
        '            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n'
        '                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n'
        '          fold = a ^ b ^ c ^ d;\n'
        '        }\n'
        '      }\n'
        '      if constexpr (RO_STORE) {\n'
        '        if ((lane & 3) == 0)\n'
        '          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n'
        '      } else {\n'
        '        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);\n'
        '      }\n'
        '      return;\n'
        '    } else {\n'
        '      float r[RT][MT][4];\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n'
        '      if constexpr (NOMMA) {\n'
        '        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n'
        '      }\n'
        '\n'
        '      const int RB = WR * RT * 16;\n'
        '      const int row0 = blockIdx.x * RB;\n'
        '      if (NORED || (RED == 0 && WK == 1)) {\n'
        '#pragma unroll\n'
        '        for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '            for (int c = 0; c < 4; ++c) {\n'
        '              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n'
        '              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n'
        '              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n'
        '                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);\n'
        '            }\n'
        '        return;\n'
        '      }\n'
        '      if constexpr (!NORED) {\n'
        '        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n'
        '        auto emit_full = [&](int idx, float4 v) {\n'
        '          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n'
        '            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n'
        '            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        auto emit_compact = [&](int idx, float4 v) {\n'
        '          if constexpr (OPT & 8) { emit_full(idx, v); return; }\n'
        '          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = 2 * t + (c & 1);\n'
        '            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n'
        '            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        if constexpr (RED == 0 || RED == 1) {\n'
        '          float4* part = reinterpret_cast<float4*>(smem);\n'
        '          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)\n'
        '              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n'
        '                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '          __syncthreads();\n'
        '          if constexpr (RED == 0) {\n'
        '            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });\n'
        '          } else {\n'
        '            float4* csum = part + (size_t)NW * RT * MT * 32;\n'
        '            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n'
        '            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n'
        '            cl.sync();\n'
        '            const int rk = (int)cl.block_rank();\n'
        '            const int per = (NI + S - 1) / S;\n'
        '            const int e1 = min(NI, (rk + 1) * per);\n'
        '            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n'
        '              float4 v = *cl.map_shared_rank(csum + idx, 0);\n'
        '              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));\n'
        '              emit_full(idx, v);\n'
        '            }\n'
        '            cl.sync();\n'
        '          }\n'
        '        } else {\n'
        '          const int NIc = WR * RT * TV * 8;\n'
        '          const int t = lane & 3, g = lane >> 2;\n'
        '          float4* recv = reinterpret_cast<float4*>(smem);\n'
        "          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.\n"
        '          if constexpr (RED == 2) {\n'
        '            smallm2_cl_wait();\n'
        '            if (t < TV) {\n'
        '              const uint32_t rbar = smallm2_mapa(bar, 0);\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n) {\n'
        '                  const int tn = n * 4 + t;\n'
        '                  if (tn < TV) {\n'
        '                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);\n'
        '                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n'
        '                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '                    if (rank == 0) *dst = v;\n'
        '                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n'
        '                  }\n'
        '                }\n'
        '            }\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            smallm2_wait_bar(bar);\n'
        '            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n'
        '          } else {\n'
        '            float4* part = recv + (size_t)S * NIc;\n'
        '            if (t < TV) {\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n)\n'
        '                  if (n * 4 + t < TV)\n'
        '                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =\n'
        '                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '            }\n'
        '            __syncthreads();\n'
        '            smallm2_cl_wait();\n'
        '            const uint32_t rbar = smallm2_mapa(bar, 0);\n'
        '            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n'
        '              float4* dst = recv + (size_t)rank * NIc + idx;\n'
        '              if (rank == 0) *dst = v;\n'
        '              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);\n'
        '            });\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            smallm2_wait_bar(bar);\n'
        '            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'static int smallm2_smem_optin() {\n'
        '  static int v = -1;\n'
        '  if (v < 0) {\n'
        '    int dev = 0;\n'
        '    cudaGetDevice(&dev);\n'
        '    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n'
        '  }\n'
        '  return v;\n'
        '}\n'
        '\n'
        '// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n'
        'static int smallm2_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  auto kern = smallm2_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;\n'
        '  if (query == 2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n'
        '    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n'
        '  }\n'
        '  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n'
        '  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n'
        '  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n'
        '  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n'
        '  if (N % RB) return -2;\n'
        '  if ((int64_t)S * WK * KU * 64 != K) return -3;\n'
        '  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n'
        '  if (m < 1 || m > MT * 8) return -5;\n'
        '  const bool cl = RED != 0;\n'
        '  if (!cl && S != 1) return -9;\n'
        '  if (cl && (S < 2 || S > 16)) return -9;\n'
        '  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);\n'
        '  size_t sm = 0;\n'
        '  if (!(DIAG & 28)) {\n'
        '    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n'
        '    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n'
        '    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n'
        '    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n'
        '  }\n'
        '  static int max_dyn = -1;\n'
        '  if (max_dyn < 0) {\n'
        '    cudaFuncAttributes fa;\n'
        '    cudaFuncGetAttributes(&fa, kern);\n'
        '    max_dyn = smallm2_smem_optin() - (int)fa.sharedSizeBytes;\n'
        '    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n'
        '    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n'
        '  }\n'
        '  if ((int64_t)sm > max_dyn) return -6;\n'
        '  cudaLaunchConfig_t lc = {};\n'
        '  cudaLaunchAttribute at[1];\n'
        '  lc.gridDim = dim3(N / RB, S, 1);\n'
        '  lc.blockDim = dim3(nt, 1, 1);\n'
        '  lc.dynamicSmemBytes = sm;\n'
        '  lc.stream = (cudaStream_t)stream;\n'
        '  if (cl) {\n'
        '    at[0].id = cudaLaunchAttributeClusterDimension;\n'
        '    at[0].val.clusterDim.x = 1;\n'
        '    at[0].val.clusterDim.y = S;\n'
        '    at[0].val.clusterDim.z = 1;\n'
        '    lc.attrs = at;\n'
        '    lc.numAttrs = 1;\n'
        '  }\n'
        '  if (query == 1) {\n'
        '    int v = 0;\n'
        '    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n'
        '                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n'
        '    cudaGetLastError();\n'
        '    return e == cudaSuccess ? v : -11;\n'
        '  }\n'
        '  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n'
        '                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n'
        '                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n'
        '                     (uint32_t)((uint64_t)u01 >> 32));\n'
        '  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n'
        '}\n'
        '\n'
        'int smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  switch (inst) {\n'
        '@@CASES@@\n'
        '    default: return -99;\n'
        '  }\n'
        '}\n'
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '\n'
        'template<int M, int NT, int ST>\n'
        '__global__ void smallm2_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n'
        '                                float* __restrict__ y, int m) {\n'
        '  constexpr int K=8192;\n'
        '  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n'
        '  float acc[M][4]={};\n'
        '  float4 ww[ST];\n'
        '  uint2 xx[ST][M];\n'
        '  auto ld=[&](int slot,int i) {\n'
        '    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) {\n'
        '      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n'
        '    }\n'
        '  };\n'
        '  #pragma unroll\n'
        '  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n'
        '  #pragma unroll 1\n'
        '  for(int base=tid;base<K/4;base+=ST*NT) {\n'
        '    #pragma unroll\n'
        '    for(int s=0;s<ST;s++) {\n'
        '      float4 v=ww[s];\n'
        '      uint2 a[M];\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) a[t]=xx[s][t];\n'
        '      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) if(t<m) {\n'
        '        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n'
        '        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n'
        '        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n'
        '        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n'
        '        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n'
        '        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n'
        '        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '  __shared__ float sums[M][NT/32];\n'
        '  #pragma unroll\n'
        '  for(int t=0;t<M;t++) if(t<m) {\n'
        '    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n'
        '    #pragma unroll\n'
        '    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '    if(lane==0) sums[t][warp]=v;\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if(warp==0) {\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) if(t<m) {\n'
        '      float v=lane<NT/32?sums[t][lane]:0.f;\n'
        '      #pragma unroll\n'
        '      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '      if(lane==0) y[t*256+row]=v;\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'int smallm2_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  auto kern=smallm2_router_scalar<1,512,2>;\n'
        '  if(query==2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n'
        '    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n'
        '  }\n'
        '  if(m!=1) return -5;\n'
        '  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n'
        '  return cudaGetLastError()==cudaSuccess?0:-8;\n'
        '}\n'
    )

    tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

    tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

    tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}

    tt_v10_sm2_SMALLM2_ESC_CAP = 32

    def tt_v10_sm2_smallm2_cuda_source(instances):
        cases = []
        for i, inst in enumerate(instances):
            args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
            cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
        return tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

    def tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
        return load_inline(name, tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

    def tt_v10_sm2_smallm2_units(w):
        """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
        return v.reshape(n // 16, k // 64, 32, 32)

    def tt_v10_sm2_smallm2_to_i16(v):
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

    def tt_v10_sm2_smallm2_to_i32(v):
        return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

    def tt_v10_sm2_smallm2_bf16_bytes(v):
        r, c = v.shape[:2]
        words = tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
        return words.view(torch.uint8).view(r, c, 2048)

    def tt_v10_sm2_smallm2_c12_bytes(v):
        """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
        r, c = v.shape[:2]
        dev = v.device
        lo = v & 255
        sign = v >> 15
        eh = v >> 8 & 127
        hist = torch.bincount(eh.flatten(), minlength=128).cpu()
        window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
        base = int(window.argmax())
        d = eh - base
        inwin = (d >= 0) & (d < 8)
        d = torch.where(inwin, d, torch.zeros_like(d))
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
        code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
        code = tt_v10_sm2_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
        lo8 = lo.to(torch.uint8)
        data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
        er, ec, el, ev = (~inwin).nonzero().unbind(1)
        xor7 = eh[er, ec, el, ev] ^ base
        return (data, base, (er, ec, el, ev, xor7))

    def tt_v10_sm2_smallm2_c12_unpack(data, base, esc):
        """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
        r, c = data.shape[:2]
        dev = data.device
        p = data.view(r, c, 3, 32, 16)
        lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
        code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
        d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
        sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
        val = (sg << 15 | base + d << 8).to(torch.int32) | lo
        er, ec, el, ev, xor7 = esc
        val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
        return val

    def tt_v10_sm2_smallm2_table(base):
        t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
        t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
        v = t0 | t1 << 32
        return v - 2 ** 64 if v >= 2 ** 63 else v

    def tt_v10_sm2_smallm2_pack_format(w, fmt):
        """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

        Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
        half 1 marks the lo matrix of a HILO pair."""
        if fmt in (2, 3):
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            mats = [tt_v10_sm2_smallm2_units(hi), tt_v10_sm2_smallm2_units(lo)]
        else:
            mats = [tt_v10_sm2_smallm2_units(w)]
        if fmt in (0, 2):
            data = torch.cat([tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
            return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
        parts, tabs, escs = ([], [], [])
        for h, v in enumerate(mats):
            data, base, esc = tt_v10_sm2_smallm2_c12_bytes(v)
            if not torch.equal(tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
                return None
            parts.append(data)
            tabs.append(tt_v10_sm2_smallm2_table(base))
            escs.append(esc + (torch.full_like(esc[0], h),))
        esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
        return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

    def tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
        if R % (wr * rt) or C != s * wk * ku:
            return None
        gx = R // (wr * rt)
        return (gx, wr * wk, gx * s * wr * wk)

    def tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
        """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
        data = pk['data']
        R, C, B = data.shape
        geo = tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
        if geo is None:
            return None
        gx, nw, G = geo
        dev = data.device
        si = torch.arange(ku, device=dev).view(ku, 1, 1)
        gw = torch.arange(G, device=dev).view(1, G, 1)
        j = torch.arange(rt, device=dev).view(1, 1, rt)
        cta, warp = (gw // nw, gw % nw)
        by, bx = (cta // gx, cta % gx)
        wri, wki = (warp % wr, warp // wr)
        tile = (bx * wr + wri) * rt + j
        chunk = (by * wk + wki) * ku + si
        idx = (tile * C + chunk).reshape(-1)
        wp = data.view(R * C, B)[idx].contiguous().view(-1)
        esc = torch.full((G, tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
        if pk['esc'] is not None and pk['esc'][0].numel():
            er, ec, el, ev, x7, half = pk['esc']
            bx_e, rem = (er // (wr * rt), er % (wr * rt))
            wr_e, j_e = (rem // rt, rem % rt)
            by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
            wk_e, s_e = (rem2 // ku, rem2 % ku)
            g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
            seq = s_e * rt + j_e
            if int(seq.max()) >= (1 << 14) - 1:
                return None
            counts = torch.bincount(g_e, minlength=G)
            if int(counts.max()) > tt_v10_sm2_SMALLM2_ESC_CAP:
                return None
            order = torch.argsort(g_e * (1 << 14) + seq)
            g_s, seq_s = (g_e[order], seq[order])
            rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
            reg = ev[order] // 2 + 16 * half[order]
            entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
            esc[g_s, rank] = tt_v10_sm2_smallm2_to_i32(entry)
        return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

    def tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
        """cfg = (WR, WK, KU, S)."""
        if sp['fmt'] == 4:
            return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
        wr, wk, ku, s = cfg
        m, k = x.shape
        return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

    def tt_v10_sm2_smallm2_rows(layer, m, n_full):
        return 1024 if layer == 'unpooler' and m <= 8 else n_full

    def tt_v10_sm2_smallm2_bf16_twin(inst):
        """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
        return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

    def tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
        """Runs every column as a one-hot input: each output must equal its weight column exactly.

        exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
        n, k = exact.shape
        mm = rows_per_launch
        dev = exact.device
        out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
        got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
        eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
        for i in range(k // mm):
            if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
                return False
        return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

    def tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
        """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
        n, k = w_rows.shape
        is_router = w_rows.dtype == torch.float32
        tol = 0.0001 if is_router else 0.01
        gen = torch.Generator(device=w_rows.device).manual_seed(seed)
        for m in ms:
            x = torch.randn((m, k), generator=gen, device=w_rows.device)
            x[m - 1].zero_()
            x[m - 1, (seed * 131 + m) % k] = 16.0
            x = x.to(torch.bfloat16)
            ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
            out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
            if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
                return False
            diff = out.double() - ref
            rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
            if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
                return False
        return True

    def tt_v10_sm2_smallm2_exact(w_rows, fmt):
        if fmt in (2, 3):
            hi = w_rows.to(torch.bfloat16)
            return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
        return w_rows

    def tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
        """Packs, stream-orders and verifies one table entry; None if any step fails."""
        m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
        rows = tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
        fmt = inst[0]
        pkey = (layer, rows, fmt)
        if pkey not in packs:
            packs[pkey] = tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
        if packs[pkey] is None:
            return None
        skey = pkey + (inst[2],) + tuple(cfg)
        if skey not in streams:
            streams[skey] = tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
        sp = streams[skey]
        if sp is None:
            return None
        vkey = ('verified', inst) + skey
        if vkey not in streams:
            w_rows = w[:rows].contiguous()
            lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
            fixed_m = inst[7] if len(inst) > 7 else 0
            ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
            streams[vkey] = tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
        if not streams[vkey]:
            return None
        return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

    def tt_v10_sm2_prepare_smallm2(weights) -> dict:
        st = {'plans': {}}
        entries = []
        for (layer, m), (inst, cfg) in tt_v10_sm2_SMALLM2_ROUTES.items():
            w = weights.get(layer)
            if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
                continue
            if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
                continue
            entries.append(((layer, 'm', m), inst, cfg))
        if not entries:
            return st
        instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
        ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
        st['ext'] = ext
        ids = {inst: i for i, inst in enumerate(instances)}
        packs, streams = ({}, {})
        for key, inst, cfg in entries:
            layer = key[0]
            w = weights[layer]
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
            if plan is None:
                plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
            if plan is not None:
                st['plans'][layer, key[2]] = plan
        return st

    def tt_v10_sm2_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
        if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
            return None
        plan = st['plans'].get((layer, x.shape[0]))
        if plan is None or n != plan['rows'] or x.shape[1] != tt_v10_sm2_SMALLM2_LAYER_K[layer]:
            return None
        if x.device != plan['wp'].device or x.data_ptr() % 16:
            return None
        out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
        rc = tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
        return out if rc == 0 else None

    tt_v10_sm2_SMALLM2_ROUTES = {('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

    def tt_v10_tune_pack_weight(w: torch.Tensor, rows: int) -> torch.Tensor:
        n, k = w.shape
        col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
        return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()

    tt_v10_tune_RL_CUDA = (
        '\n'
        '#include <cuda.h>\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#define OD __device__ __forceinline__\n'
        'OD uint32_t rl_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        'OD void rl_wait(uint32_t b,int ph) {\n'
        '  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");\n'
        '}\n'
        'OD void rl_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\n'
        'OD uint64_t rl_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\n'
        'OD void rl_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {\n'
        '  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");\n'
        '}\n'
        '@@MMA@@\n'
        '\n'
        'template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n'
        '__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void rl_early_projection(\n'
        '    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,\n'
        '    const uint8_t* w,__nv_bfloat16* out,int m) {\n'
        '  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;\n'
        '  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);\n'
        '  extern __shared__ __align__(1024) uint8_t sm[];\n'
        '  uint32_t base=rl_addr(sm),full=base+ST*SB,empty=full+ST*8;\n'
        '  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;\n'
        '  uint64_t pw,px;\n'
        '  if constexpr(HINT) {\n'
        '    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));\n'
        '    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));\n'
        '  } else {\n'
        '    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;\n'
        '  }\n'
        '  auto load=[&](int i) {\n'
        '    int s=i%ST;\n'
        '    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");\n'
        '    if constexpr(PACK==1) {\n'
        '      const uint8_t* src=w+((long)nt*IT+i)*AB;\n'
        '      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");\n'
        '    } else if constexpr(PACK==2) rl_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);\n'
        '    else rl_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);\n'
        '    rl_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n'
        '  };\n'
        '  if(tid==((PIPE&2)?CT:0)) {\n'
        '    #pragma unroll\n'
        '    for(int s=0;s<ST;s++) {\n'
        '      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");\n'
        '      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");\n'
        '    }\n'
        '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n'
        '  }\n'
        '  if constexpr(PIPE&4) __syncthreads();\n'
        '  if(tid==((PIPE&2)?CT:0)) {\n'
        '    #pragma unroll\n'
        '    for(int i=0;i<ST && i<IT;i++) load(i);\n'
        '  }\n'
        '  if constexpr(!(PIPE&4)) __syncthreads();\n'
        '  if((PIPE&2) && tid>=CT) {\n'
        '    if(tid==CT) {\n'
        '      #pragma unroll 1\n'
        '      for(int i=ST;i<IT;i++) {rl_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n'
        '    }\n'
        '  } else {\n'
        '    float a[BN/2];\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");\n'
        '    #pragma unroll 1\n'
        '    for(int i=0;i<IT;i++) {\n'
        '      int s=i%ST;\n'
        '      rl_wait(full+s*8,(i/ST)&1);\n'
        '      #pragma unroll\n'
        '      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n'
        '      #pragma unroll\n'
        '      for(int kk=0;kk<BK/16;kk++) {\n'
        '        uint64_t aw=rl_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);\n'
        '        uint64_t ax=rl_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);\n'
        '        @@CALL@@\n'
        '      }\n'
        '      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");\n'
        '      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");\n'
        '      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n'
        '      #pragma unroll\n'
        '      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '      int done=i-((PIPE&1)?1:0);\n'
        '      if(done>=0 && done+ST<IT && tid%128==0) {\n'
        '        if constexpr(PIPE&2) rl_arrive(empty+(done%ST)*8);\n'
        '        else load(done+ST);\n'
        '      }\n'
        '    }\n'
        '    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");\n'
        '    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");\n'
        '    #pragma unroll\n'
        '    for(int j=0;j<BN/8;j++) {\n'
        '      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);\n'
        '      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n'
        '      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);\n'
        '      uint32_t d=base+2*(cb*72+rb);\n'
        '      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");\n'
        '    }\n'
        '  }\n'
        '  __syncthreads();\n'
        '  for(int q=tid;q<T*(R/8);q+=TH) {\n'
        '    int token=q/(R/8),row=(q%(R/8))*8;\n'
        '    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);\n'
        '  }\n'
        '}\n'
        '\n'
        'using OLEncode=decltype(&cuTensorMapEncodeTiled);\n'
        'int rl_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {\n'
        '  static OLEncode encode=nullptr;\n'
        '  if(!encode) {\n'
        '    cudaDriverEntryPointQueryResult q;\n'
        '    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);\n'
        '    if(rc!=cudaSuccess || !encode) return -100;\n'
        '  }\n'
        '  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};\n'
        '  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};\n'
        '  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n'
        '}\n'
        'template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n'
        'int rl_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n'
        '  constexpr int T=BN*TW,R=PACK==2?32:64;\n'
        '  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;\n'
        '  auto fn=rl_early_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;\n'
        '  static bool configured=false;\n'
        '  if(!configured) {\n'
        '    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;\n'
        '    configured=true;\n'
        '  }\n'
        '  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}\n'
        '  CUtensorMap tw,tx;\n'
        '  int rc=rl_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n'
        '  rc=rl_map(&tx,(void*)x,m,T,BK);if(rc)return rc;\n'
        '  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);\n'
        '  return int(cudaGetLastError());\n'
        '}\n'
        'int rl_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {\n'
        '  switch(id) {@@CASES@@ default:return -99;}\n'
        '}\n'
    )

    tt_v10_tune_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

    def tt_v10_tune_rl_source(configs) -> str:
        ns = sorted({c[0] for c in configs})
        functions = []
        for n in ns:
            regs = ','.join((f'%{i}' for i in range(n // 2)))
            outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
            functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
        cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
        calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
        return tt_v10_tune_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

    def tt_v10_tune_build_rl(configs):
        for cfg in configs:
            bn, bk, st, pipe, pack, hint, *groups = cfg
            tw = groups[0] if groups else 1
            if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
                raise ValueError(f'unsupported rl configuration: {cfg}')
        tt_v10_tune_source = tt_v10_tune_rl_source(configs)
        name = 'rl_' + hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
        return load_inline(name, tt_v10_tune_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

    def prepare_v10_tune(weights: Weights) -> dict:
        weight = weights.get('o')
        if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (8192, 1024):
            return {}
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            return {}
        return {'ext': tt_v10_tune_build_rl([(256, 64, 5, 7, 1, 1, 1)]),
                'weight': tt_v10_tune_pack_weight(weight.contiguous(), 64)}

    def run_v10_tune(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor | None:
        if layer != 'o' or n != 8192 or not state or x.ndim != 2 or tuple(x.shape) != (256, 1024):
            return None
        if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16 or x.device != state['weight'].device:
            return None
        output = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
        rc = state['ext'].rl_launch(0, x.data_ptr(), state['weight'].data_ptr(), output.data_ptr(),
                                    256, torch.cuda.current_stream(x.device).cuda_stream, 0)
        if rc:
            raise RuntimeError(f'm256tune O launch failed: {rc}')
        return output

    tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE = (
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#include <type_traits>\n'
        '#include <cooperative_groups.h>\n'
        '\n'
        '#define SMD __device__ __forceinline__\n'
        '\n'
        'SMD int4 smtrace_ldw(const int4* p) {\n'
        '  int4 v;\n'
        '  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n'
        '               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMD uint4 smtrace_ldx(const void* p) {\n'
        '  uint4 v;\n'
        '  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMD uint32_t smtrace_lde(const uint32_t* p) {\n'
        '  uint32_t v;\n'
        '  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SMD uint32_t smtrace_prmt(uint32_t a, uint32_t b, uint32_t s) {\n'
        '  uint32_t d;\n'
        '  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SMD void smtrace_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n'
        '  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n'
        '      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n'
        '}\n'
        '\n'
        '// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\n'
        'SMD void smtrace_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n'
        '  const uint32_t cm = C & 0x77777777u;\n'
        '  const uint32_t hlo = smtrace_prmt(T0, T1, cm);\n'
        '  const uint32_t hhi = smtrace_prmt(T0, T1, cm >> 16);\n'
        '  a[0] = smtrace_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n'
        '  a[1] = smtrace_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n'
        '  a[2] = smtrace_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n'
        '  a[3] = smtrace_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n'
        '}\n'
        '\n'
        '__device__ __noinline__ void smtrace_never(void* p) { asm volatile("trap;"); }\n'
        '\n'
        '// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\n'
        'SMD void smtrace_order_point(int m, void* p) {\n'
        '  uint32_t t;\n'
        '  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n'
        '  if ((t & (uint32_t)m) == 0x7fffffffu) smtrace_never(p);\n'
        '}\n'
        '\n'
        'SMD uint32_t smtrace_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        '\n'
        'SMD uint32_t smtrace_rank() {\n'
        '  uint32_t r;\n'
        '  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n'
        '  return r;\n'
        '}\n'
        '\n'
        'SMD uint32_t smtrace_mapa(uint32_t a, uint32_t rank) {\n'
        '  uint32_t d;\n'
        '  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SMD void smtrace_mbar_init(uint32_t bar, uint32_t count) {\n'
        '  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n'
        '}\n'
        '\n'
        'SMD void smtrace_mbar_expect(uint32_t bar, uint32_t tx) {\n'
        '  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n'
        '}\n'
        '\n'
        'SMD void smtrace_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\n'
        'SMD void smtrace_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\n'
        'SMD void smtrace_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n'
        '\n'
        'SMD void smtrace_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n'
        '               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n'
        '}\n'
        '\n'
        'SMD bool smtrace_mbar_try(uint32_t bar) {\n'
        '  uint32_t ok;\n'
        '  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n'
        '               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n'
        '  return ok != 0;\n'
        '}\n'
        '\n'
        'SMD uint64_t smtrace_gtime() {\n'
        '  uint64_t t;\n'
        '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n'
        '  return t;\n'
        '}\n'
        '\n'
        '// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\n'
        'SMD void smtrace_wait_bar(uint32_t bar) {\n'
        '  if (smtrace_mbar_try(bar)) return;\n'
        '  const uint64_t t0 = smtrace_gtime();\n'
        '  while (!smtrace_mbar_try(bar))\n'
        '    if (smtrace_gtime() - t0 > 20000000ull) __trap();\n'
        '}\n'
        '\n'
        'SMD void smtrace_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n'
        '\n'
        'SMD float4 smtrace_shfl_xor4(const float4& v, int o) {\n'
        '  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n'
        '                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n'
        '}\n'
        '\n'
        'SMD void smtrace_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\n'
        'SMD void smtrace_store(float* p, float v) { *p = v; }\n'
        '\n'
        'template <int FMT> struct SmFmt;\n'
        'template <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };\n'
        'template <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };\n'
        'template <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };\n'
        'template <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };\n'
        '\n'
        'template <int FMT>\n'
        'SMD void smtrace_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n'
        '                   uint32_t U0, uint32_t U1) {\n'
        '  if constexpr (FMT == 0 || FMT == 2) {\n'
        '#pragma unroll\n'
        '    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {\n'
        '      A[4 * l + 0] = (uint32_t)w[l].x;\n'
        '      A[4 * l + 1] = (uint32_t)w[l].y;\n'
        '      A[4 * l + 2] = (uint32_t)w[l].z;\n'
        '      A[4 * l + 3] = (uint32_t)w[l].w;\n'
        '    }\n'
        '  } else {\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {\n'
        '      const int4* u = w + 3 * h;\n'
        '      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n'
        '      smtrace_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n'
        '      smtrace_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n'
        '      smtrace_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n'
        '      smtrace_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        "// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.\n"
        'template <int NR>\n'
        'SMD void smtrace_patch(uint32_t (&A)[NR], uint32_t e) {\n'
        '  const uint32_t reg = (e >> 8) & 31u;\n'
        '  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n'
        '#pragma unroll\n'
        '  for (int r = 0; r < NR; ++r)\n'
        '    if (reg == (uint32_t)r) A[r] ^= xr;\n'
        '}\n'
        '\n'
        "// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.\n"
        'template <class Emit>\n'
        'SMD void smtrace_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n'
        '  for (int idx = tid; idx < NI; idx += nt) {\n'
        '    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n'
        '    for (int p = 0; p < P; ++p) smtrace_add4(v, src[(size_t)p * PSTR + idx]);\n'
        '    emit(idx, v);\n'
        '  }\n'
        '}\n'
        '\n'
        '// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n'
        '// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n'
        '// straight-line code block at the end, 128 (with 4) no output stores.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>\n'
        '__global__ void __launch_bounds__(NTMAX) smtrace_gemv(\n'
        '    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n'
        '    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n'
        '    uint32_t U0, uint32_t U1) {\n'
        '  const int m = CM ? CM : m_arg;\n'
        '  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n'
        '  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n'
        '  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;\n'
        '  constexpr bool C12 = FMT == 1 || FMT == 3;\n'
        '  constexpr bool HILO = FMT >= 2;\n'
        '  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n'
        '  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n'
        '  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n'
        '  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n'
        '  OutT* out = reinterpret_cast<OutT*>(outp);\n'
        '  extern __shared__ __align__(16) unsigned char smem[];\n'
        '  __shared__ __align__(8) uint64_t s_bar;\n'
        '  if constexpr (EMPTY) {\n'
        '    return;\n'
        '  } else {\n'
        '    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n'
        '    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n'
        '    const int S = CS ? CS : gridDim.y;\n'
        '    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n'
        '    const int G = GX * S * NW;\n'
        '    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n'
        '    const int rt0 = (blockIdx.x * WR + wr) * RT;\n'
        '    const int c0 = (blockIdx.y * WK + wk) * KU;\n'
        '    const int TV = min(4 * MT, (m + 1) >> 1);\n'
        '    const uint32_t bar = smtrace_smem_u32(&s_bar);\n'
        '    uint32_t rank = 0;\n'
        '    if constexpr (PUSH) {\n'
        '      rank = smtrace_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        smtrace_mbar_init(bar, 1);\n'
        '        smtrace_fence_init();\n'
        '      }\n'
        '      if (rank == 0) {\n'
        '        __syncthreads();\n'
        '        if(tid==0) {\n'
        '          const uint32_t parts=RED==2?(uint32_t)((S-1)*NW):(uint32_t)((S-1)*WR);\n'
        '          smtrace_mbar_expect(bar,parts*RT*TV*128u);\n'
        '        }\n'
        '      }\n'
        '      smtrace_cl_arrive();\n'
        '    }\n'
        '\n'
        '    uint32_t my_e = 0xFFFFFFFFu;\n'
        '    if constexpr (C12 && !RO) my_e = smtrace_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n'
        '    const size_t sstride = (size_t)G * (RT * NL * 32);\n'
        '    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n'
        '\n'
        '    int4 w[ST][RT][NL];\n'
        '    uint4 xv[ST][MT][2];\n'
        '    auto load = [&](int slot, int s) {\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int l = 0; l < NL; ++l) w[slot][j][l] = smtrace_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n'
        '      if constexpr (!NOX && !RO) {\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n) {\n'
        '          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n'
        '          xv[slot][n][0] = smtrace_ldx(xp);\n'
        '          xv[slot][n][1] = smtrace_ldx(xp + 8);\n'
        '        }\n'
        '      }\n'
        '    };\n'
        '#pragma unroll\n'
        '    for (int s = 0; s < ST; ++s) load(s, s);\n'
        '    smtrace_order_point(m, outp);\n'
        '\n'
        '    float acc[NA][RT][MT][4];\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < NA; ++h)\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n'
        '    uint32_t fold = 0;\n'
        '    uint32_t nxt = 0xFFFFFFFFu;\n'
        '    int ep = 0;\n'
        '    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n'
        '\n'
        '#pragma unroll 1\n'
        '    for (int i = 0; i < KU; i += ST) {\n'
        '#pragma unroll\n'
        '      for (int s = 0; s < ST; ++s) {\n'
        '        const int si = i + s;\n'
        '        if (ST == 3 && si >= KU) break;\n'
        '        if constexpr (RO) {\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int l = 0; l < NL; ++l)\n'
        '              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          smtrace_order_point(m, outp);\n'
        '        } else {\n'
        '          uint32_t A[RT][NA * 16];\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j) smtrace_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n'
        '          uint32_t b[MT][8];\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n) {\n'
        '            if constexpr (NOX) {\n'
        '#pragma unroll\n'
        '              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n'
        '            } else {\n'
        '              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n'
        '              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n'
        '            }\n'
        '          }\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          smtrace_order_point(m, outp);\n'
        '          if constexpr (C12) {\n'
        '#pragma unroll\n'
        '            for (int j = 0; j < RT; ++j) {\n'
        '              const uint32_t useq = (uint32_t)(si * RT + j);\n'
        '              if ((nxt >> 18) == useq) {\n'
        '                do {\n'
        '                  if (lane == (int)((nxt >> 13) & 31u)) smtrace_patch<NA * 16>(A[j], nxt);\n'
        '                  ++ep;\n'
        '                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n'
        '                } while ((nxt >> 18) == useq);\n'
        '              }\n'
        '            }\n'
        '          }\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '              for (int q = 0; q < 4; ++q) {\n'
        '                if constexpr (NOMMA) {\n'
        '#pragma unroll\n'
        '                  for (int h = 0; h < NA; ++h)\n'
        '                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n'
        '                            A[j][16 * h + 4 * q + 3];\n'
        '                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n'
        '                } else {\n'
        '                  smtrace_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                  if constexpr (HILO) smtrace_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                }\n'
        '              }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '\n'
        '    if constexpr (RO) {\n'
        '      if constexpr (RO_CODE) {\n'
        '        if (warp == 0) {\n'
        '          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n'
        '          const uint32_t k = (uint32_t)m | 1u;\n'
        '#pragma unroll\n'
        '          for (int q = 0; q < 64; ++q)\n'
        '            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n'
        '                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n'
        '          fold = a ^ b ^ c ^ d;\n'
        '        }\n'
        '      }\n'
        '      if constexpr (RO_STORE) {\n'
        '        if ((lane & 3) == 0)\n'
        '          smtrace_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n'
        '      } else {\n'
        '        if (fold == 0x9e3779b9u) smtrace_store(out, 1.f);\n'
        '      }\n'
        '      return;\n'
        '    } else {\n'
        '      float r[RT][MT][4];\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n'
        '      if constexpr (NOMMA) {\n'
        '        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n'
        '      }\n'
        '\n'
        '      const int RB = WR * RT * 16;\n'
        '      const int row0 = blockIdx.x * RB;\n'
        '      if (NORED || (RED == 0 && WK == 1)) {\n'
        '#pragma unroll\n'
        '        for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '            for (int c = 0; c < 4; ++c) {\n'
        '              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n'
        '              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n'
        '              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n'
        '                smtrace_store(out + (size_t)tok * N + row, r[j][n][c]);\n'
        '            }\n'
        '        return;\n'
        '      }\n'
        '      if constexpr (!NORED) {\n'
        '        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n'
        '        auto emit_full = [&](int idx, float4 v) {\n'
        '          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n'
        '            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n'
        '            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        auto emit_compact = [&](int idx, float4 v) {\n'
        '          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = 2 * t + (c & 1);\n'
        '            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n'
        '            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        if constexpr (RED == 0 || RED == 1) {\n'
        '          float4* part = reinterpret_cast<float4*>(smem);\n'
        '          const int NI = WR * RT * MT * 32;\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n'
        '                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '          __syncthreads();\n'
        '          if constexpr (RED == 0) {\n'
        '            smtrace_reduce(part, NI, WK, NI, tid, nt, emit_full);\n'
        '          } else {\n'
        '            float4* csum = part + (size_t)NW * RT * MT * 32;\n'
        '            smtrace_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n'
        '            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n'
        '            cl.sync();\n'
        '            const int rk = (int)cl.block_rank();\n'
        '            const int per = (NI + S - 1) / S;\n'
        '            const int e1 = min(NI, (rk + 1) * per);\n'
        '            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n'
        '              float4 v = *cl.map_shared_rank(csum + idx, 0);\n'
        '              for (int q = 1; q < S; ++q) smtrace_add4(v, *cl.map_shared_rank(csum + idx, q));\n'
        '              emit_full(idx, v);\n'
        '            }\n'
        '            cl.sync();\n'
        '          }\n'
        '        } else {\n'
        '          const int NIc = WR * RT * TV * 8;\n'
        '          const int t = lane & 3, g = lane >> 2;\n'
        '          float4* recv = reinterpret_cast<float4*>(smem);\n'
        "          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.\n"
        '          if constexpr (RED == 2) {\n'
        '            smtrace_cl_wait();\n'
        '            if (t < TV) {\n'
        '              const uint32_t rbar = smtrace_mapa(bar, 0);\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n) {\n'
        '                  const int tn = n * 4 + t;\n'
        '                  if (tn < TV) {\n'
        '                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;\n'
        '                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n'
        '                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '                    if (rank == 0) *dst = v;\n'
        '                    else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);\n'
        '                  }\n'
        '                }\n'
        '            }\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            smtrace_wait_bar(bar);\n'
        '            smtrace_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n'
        '          } else {\n'
        '            float4* part = recv + (size_t)S * NIc;\n'
        '            if (t < TV) {\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n)\n'
        '                  if (n * 4 + t < TV)\n'
        '                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =\n'
        '                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '            }\n'
        '            __syncthreads();\n'
        '            smtrace_cl_wait();\n'
        '            const uint32_t rbar = smtrace_mapa(bar, 0);\n'
        '            smtrace_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n'
        '              float4* dst = recv + (size_t)rank * NIc + idx;\n'
        '              if (rank == 0) *dst = v;\n'
        '              else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);\n'
        '            });\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            smtrace_wait_bar(bar);\n'
        '            smtrace_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'static int smtrace_smem_optin() {\n'
        '  static int v = -1;\n'
        '  if (v < 0) {\n'
        '    int dev = 0;\n'
        '    cudaGetDevice(&dev);\n'
        '    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n'
        '  }\n'
        '  return v;\n'
        '}\n'
        '\n'
        '// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>\n'
        'static int smtrace_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  auto kern = smtrace_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;\n'
        '  if (query == 2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n'
        '    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n'
        '  }\n'
        '  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n'
        '  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n'
        '  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n'
        '  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n'
        '  if (N % RB) return -2;\n'
        '  if ((int64_t)S * WK * KU * 64 != K) return -3;\n'
        '  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n'
        '  if (m < 1 || m > MT * 8) return -5;\n'
        '  const bool cl = RED != 0;\n'
        '  if (!cl && S != 1) return -9;\n'
        '  if (cl && (S < 2 || S > 16)) return -9;\n'
        '  const int TV = std::min(4 * MT, (m + 1) / 2);\n'
        '  size_t sm = 0;\n'
        '  if (!(DIAG & 28)) {\n'
        '    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n'
        '    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n'
        '    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n'
        '    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n'
        '  }\n'
        '  static int max_dyn = -1;\n'
        '  if (max_dyn < 0) {\n'
        '    cudaFuncAttributes fa;\n'
        '    cudaFuncGetAttributes(&fa, kern);\n'
        '    max_dyn = smtrace_smem_optin() - (int)fa.sharedSizeBytes;\n'
        '    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n'
        '    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n'
        '  }\n'
        '  if ((int64_t)sm > max_dyn) return -6;\n'
        '  cudaLaunchConfig_t lc = {};\n'
        '  cudaLaunchAttribute at[1];\n'
        '  lc.gridDim = dim3(N / RB, S, 1);\n'
        '  lc.blockDim = dim3(nt, 1, 1);\n'
        '  lc.dynamicSmemBytes = sm;\n'
        '  lc.stream = (cudaStream_t)stream;\n'
        '  if (cl) {\n'
        '    at[0].id = cudaLaunchAttributeClusterDimension;\n'
        '    at[0].val.clusterDim.x = 1;\n'
        '    at[0].val.clusterDim.y = S;\n'
        '    at[0].val.clusterDim.z = 1;\n'
        '    lc.attrs = at;\n'
        '    lc.numAttrs = 1;\n'
        '  }\n'
        '  if (query == 1) {\n'
        '    int v = 0;\n'
        '    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n'
        '                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n'
        '    cudaGetLastError();\n'
        '    return e == cudaSuccess ? v : -11;\n'
        '  }\n'
        '  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n'
        '                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n'
        '                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n'
        '                     (uint32_t)((uint64_t)u01 >> 32));\n'
        '  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n'
        '}\n'
        '\n'
        'int smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  switch (inst) {\n'
        '@@CASES@@\n'
        '    default: return -99;\n'
        '  }\n'
        '}\n'
    )

    tt_v13_smtrace8_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

    tt_v13_smtrace8_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

    tt_v13_smtrace8_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}

    tt_v13_smtrace8_SMTRACE_ESC_CAP = 32

    def tt_v13_smtrace8_smtrace_cuda_source(instances):
        cases = []
        for i, inst in enumerate(instances):
            args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
            cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
        return tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

    def tt_v13_smtrace8_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
        return load_inline(name, tt_v13_smtrace8_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v13_smtrace8_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

    def tt_v13_smtrace8_smtrace_units(w):
        """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
        return v.reshape(n // 16, k // 64, 32, 32)

    def tt_v13_smtrace8__smtrace_to_i16(v):
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

    def tt_v13_smtrace8__smtrace_to_i32(v):
        return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

    def tt_v13_smtrace8_smtrace_bf16_bytes(v):
        r, c = v.shape[:2]
        words = tt_v13_smtrace8__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
        return words.view(torch.uint8).view(r, c, 2048)

    def tt_v13_smtrace8_smtrace_c12_bytes(v):
        """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
        r, c = v.shape[:2]
        dev = v.device
        lo = v & 255
        sign = v >> 15
        eh = v >> 8 & 127
        hist = torch.bincount(eh.flatten(), minlength=128).cpu()
        window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
        base = int(window.argmax())
        d = eh - base
        inwin = (d >= 0) & (d < 8)
        d = torch.where(inwin, d, torch.zeros_like(d))
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
        code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
        code = tt_v13_smtrace8__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
        lo8 = lo.to(torch.uint8)
        data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
        er, ec, el, ev = (~inwin).nonzero().unbind(1)
        xor7 = eh[er, ec, el, ev] ^ base
        return (data, base, (er, ec, el, ev, xor7))

    def tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc):
        """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
        r, c = data.shape[:2]
        dev = data.device
        p = data.view(r, c, 3, 32, 16)
        lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
        code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
        d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
        sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
        val = (sg << 15 | base + d << 8).to(torch.int32) | lo
        er, ec, el, ev, xor7 = esc
        val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
        return val

    def tt_v13_smtrace8_smtrace_table(base):
        t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
        t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
        v = t0 | t1 << 32
        return v - 2 ** 64 if v >= 2 ** 63 else v

    def tt_v13_smtrace8_smtrace_pack_format(w, fmt):
        """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

        Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
        half 1 marks the lo matrix of a HILO pair."""
        if fmt in (2, 3):
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            mats = [tt_v13_smtrace8_smtrace_units(hi), tt_v13_smtrace8_smtrace_units(lo)]
        else:
            mats = [tt_v13_smtrace8_smtrace_units(w)]
        if fmt in (0, 2):
            data = torch.cat([tt_v13_smtrace8_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
            return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
        parts, tabs, escs = ([], [], [])
        for h, v in enumerate(mats):
            data, base, esc = tt_v13_smtrace8_smtrace_c12_bytes(v)
            if not torch.equal(tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc), v):
                return None
            parts.append(data)
            tabs.append(tt_v13_smtrace8_smtrace_table(base))
            escs.append(esc + (torch.full_like(esc[0], h),))
        esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
        return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

    def tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s):
        if R % (wr * rt) or C != s * wk * ku:
            return None
        gx = R // (wr * rt)
        return (gx, wr * wk, gx * s * wr * wk)

    def tt_v13_smtrace8_smtrace_stream(pk, rt, wr, wk, ku, s):
        """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
        data = pk['data']
        R, C, B = data.shape
        geo = tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s)
        if geo is None:
            return None
        gx, nw, G = geo
        dev = data.device
        si = torch.arange(ku, device=dev).view(ku, 1, 1)
        gw = torch.arange(G, device=dev).view(1, G, 1)
        j = torch.arange(rt, device=dev).view(1, 1, rt)
        cta, warp = (gw // nw, gw % nw)
        by, bx = (cta // gx, cta % gx)
        wri, wki = (warp % wr, warp // wr)
        tile = (bx * wr + wri) * rt + j
        chunk = (by * wk + wki) * ku + si
        idx = (tile * C + chunk).reshape(-1)
        wp = data.view(R * C, B)[idx].contiguous().view(-1)
        esc = torch.full((G, tt_v13_smtrace8_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
        if pk['esc'] is not None and pk['esc'][0].numel():
            er, ec, el, ev, x7, half = pk['esc']
            bx_e, rem = (er // (wr * rt), er % (wr * rt))
            wr_e, j_e = (rem // rt, rem % rt)
            by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
            wk_e, s_e = (rem2 // ku, rem2 % ku)
            g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
            seq = s_e * rt + j_e
            if int(seq.max()) >= (1 << 14) - 1:
                return None
            counts = torch.bincount(g_e, minlength=G)
            if int(counts.max()) > tt_v13_smtrace8_SMTRACE_ESC_CAP:
                return None
            order = torch.argsort(g_e * (1 << 14) + seq)
            g_s, seq_s = (g_e[order], seq[order])
            rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
            reg = ev[order] // 2 + 16 * half[order]
            entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
            esc[g_s, rank] = tt_v13_smtrace8__smtrace_to_i32(entry)
        return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

    def tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
        """cfg = (WR, WK, KU, S)."""
        wr, wk, ku, s = cfg
        m, k = x.shape
        return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

    tt_v13_smtrace8_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)), ('router', 8): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)), ('qkv_gate', 8): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('pooler', 8): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('o', 8): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('unpooler', 8): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2))}

    tt_v13_smtrace8_SMTRACE_MAX_M = 32

    tt_v13_smtrace8_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)), ('pooler', 2): ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 2): ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4)), ('unpooler', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))}

    def tt_v13_smtrace8_smtrace_mt(m):
        return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

    def tt_v13_smtrace8_smtrace_rows(layer, m, n_full):
        return 1024 if layer == 'unpooler' and m <= 8 else n_full

    def tt_v13_smtrace8_smtrace_bf16_twin(inst):
        """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
        return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

    def tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
        """Runs every column as a one-hot input: each output must equal its weight column exactly.

        exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
        n, k = exact.shape
        mm = rows_per_launch
        dev = exact.device
        out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
        got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
        eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
        for i in range(k // mm):
            if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
                return False
        return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

    def tt_v13_smtrace8_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
        """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
        n, k = w_rows.shape
        is_router = w_rows.dtype == torch.float32
        tol = 0.0001 if is_router else 0.01
        gen = torch.Generator(device=w_rows.device).manual_seed(seed)
        for m in ms:
            x = torch.randn((m, k), generator=gen, device=w_rows.device)
            x[m - 1].zero_()
            x[m - 1, (seed * 131 + m) % k] = 16.0
            x = x.to(torch.bfloat16)
            ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
            out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
            if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
                return False
            diff = out.double() - ref
            rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
            if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
                return False
        return True

    def tt_v13_smtrace8_smtrace_exact(w_rows, fmt):
        if fmt in (2, 3):
            hi = w_rows.to(torch.bfloat16)
            return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
        return w_rows

    def tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
        """Packs, stream-orders and verifies one table entry; None if any step fails."""
        m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
        rows = tt_v13_smtrace8_smtrace_rows(layer, m_hi, w.shape[0])
        fmt = inst[0]
        pkey = (layer, rows, fmt)
        if pkey not in packs:
            packs[pkey] = tt_v13_smtrace8_smtrace_pack_format(w[:rows].contiguous(), fmt)
        if packs[pkey] is None:
            return None
        skey = pkey + (inst[2],) + tuple(cfg)
        if skey not in streams:
            streams[skey] = tt_v13_smtrace8_smtrace_stream(packs[pkey], inst[2], *cfg)
        sp = streams[skey]
        if sp is None:
            return None
        vkey = ('verified', inst) + skey
        if vkey not in streams:
            w_rows = w[:rows].contiguous()
            lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
            fixed_m = inst[7] if len(inst) > 7 else 0
            ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
            streams[vkey] = tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v13_smtrace8_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v13_smtrace8_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
        if not streams[vkey]:
            return None
        return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

    def tt_v13_smtrace8_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
        table_m = dict(tt_v13_smtrace8_SMTRACE_TABLE_M if table_m is None else table_m)
        table_mt = dict(tt_v13_smtrace8_SMTRACE_TABLE_MT if table_mt is None else table_mt)
        entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
        specialized = []
        for key, (inst, cfg) in entries:
            if len(inst) == 7:
                layer, kind, m = key
                fixed_m = m if kind == 'm' else 0
                rows = tt_v13_smtrace8_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
                inst = inst + (fixed_m, rows, tt_v13_smtrace8_SMTRACE_LAYER_K[layer], *cfg)
            specialized.append((key, (inst, cfg)))
        entries = specialized
        instances = []
        for _, (inst, _) in entries:
            instances.append(inst)
            if tt_v13_smtrace8_smtrace_bf16_twin(inst):
                instances.append(tt_v13_smtrace8_smtrace_bf16_twin(inst))
        instances = list(dict.fromkeys(instances))
        st = {'plans': {}}
        if not instances:
            return st
        ext = tt_v13_smtrace8_smtrace_build(instances, name)
        st['ext'] = ext
        inst_id = {inst: i for i, inst in enumerate(instances)}
        packs, streams = ({}, {})
        for key, (inst, cfg) in sorted(entries):
            layer = key[0]
            plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
            if plan is None and tt_v13_smtrace8_smtrace_bf16_twin(inst):
                plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v13_smtrace8_smtrace_bf16_twin(inst), cfg)
            if plan is not None:
                st['plans'][key] = plan
        del packs
        return st

    def tt_v13_smtrace8_smtrace_run_core(x, layer, n, st):
        m = x.shape[0]
        if not 1 <= m <= tt_v13_smtrace8_SMTRACE_MAX_M:
            return None
        sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v13_smtrace8_smtrace_mt(m)))
        if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v13_smtrace8_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
            return None
        x = x.contiguous()
        if x.data_ptr() % 16:
            return None
        out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
        if tt_v13_smtrace8_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
            return None
        return out

    tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

    def tt_v13_smtrace8_prepare_smtrace(weights):
        usable = {}
        for (layer, m), (inst, cfg) in tt_v13_smtrace8_SMTRACE_ROUTES.items():
            w = weights.get(layer)
            if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
                usable[layer] = w
        routes = {key: value for key, value in tt_v13_smtrace8_SMTRACE_ROUTES.items() if key[0] in usable}
        return tt_v13_smtrace8_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

    def tt_v13_smtrace8_run_smtrace(x, layer, n, st):
        if (layer, x.shape[0]) not in tt_v13_smtrace8_SMTRACE_ROUTES:
            return None
        return tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)

    def tt_v14_qkv_qbig_mma(n):
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'

    tt_v14_qkv_QKVBIG_CUDA = (
        '\n'
        '#include <cuda.h>\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#define QD __device__ __forceinline__\n'
        'QD uint32_t q_addr(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}\n'
        'QD void q_init(uint32_t p,int n) {asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}\n'
        'QD void q_expect(uint32_t p,int n) {asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(p),"r"(n):"memory");}\n'
        'QD void q_wait(uint32_t p,int phase) {\n'
        '  asm volatile("{.reg .pred p; L: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p,[%0],%1; @!p bra L;}"::"r"(p),"r"(phase):"memory");\n'
        '}\n'
        'QD void q_sync() {asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;":::"memory");}\n'
        'QD uint32_t q_map(uint32_t p,int rank) {uint32_t d;asm("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(d):"r"(p),"r"(rank));return d;}\n'
        'QD void q_push(uint32_t p,float4 v,uint32_t b) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0],{%1,%2,%3,%4},[%5];"::"r"(p),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n'
        '}\n'
        'QD uint64_t q_desc(uint32_t p) {return uint64_t((p&0x3ffff)>>4)|(uint64_t(64)<<32)|(uint64_t(1)<<62);}\n'
        'QD uint64_t q_time() {uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}\n'
        'QD void q_tma(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n'
        '  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k),"r"(r),"r"(b),"l"(policy):"memory");\n'
        '}\n'
        'QD void q_tma3(uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n'
        '  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"::"r"(p),"l"(tm),"r"(k/64),"r"(r),"r"(b),"l"(policy):"memory");\n'
        '}\n'
        '@@MMA@@\n'
        'template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>\n'
        '__global__ __launch_bounds__((PIPE&2)?256:128) void qb_kernel(const __grid_constant__ CUtensorMap tw,\n'
        '    const __grid_constant__ CUtensorMap tx,__nv_bfloat16* out,int m,unsigned long long* stamps) {\n'
        '  constexpr int KT=(PIPE&32)?2:1, LR=(2304+G-1)/G, WB=NR*128, XB=TM*128, SB=KT*(WB+XB);\n'
        '  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);\n'
        '  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;\n'
        '  extern __shared__ __align__(1024) unsigned char sm[];\n'
        '  const int tid=threadIdx.x, lane=tid&31, rank=blockIdx.x%S, group=blockIdx.x/S;\n'
        '  const int tok=blockIdx.y*TM, row=group*2304/G, rows=(group+1)*2304/G-row;\n'
        '  const int kb=(128/KT)*rank/S, nkb=(128/KT)*(rank+1)/S-kb, own=RED>=4?((NV/128+S-1-rank)/S)*128:min(OWN,NV-rank*OWN);\n'
        '  const uint32_t base=q_addr(sm), bars=base+ST*SB, empty=bars+ST*8, rb=base+RB;\n'
        '  uint64_t begin=0, stream_end=0;\n'
        '  if constexpr(PROFILE) begin=q_time();\n'
        '  if constexpr(LR<NR) {\n'
        '    for(int i=tid;i<ST*(NR-LR)*32;i+=blockDim.x) {\n'
        '      int stage=i/((NR-LR)*32), j=i%((NR-LR)*32);\n'
        '      ((uint32_t*)(sm+stage*SB+LR*128))[j]=0;\n'
        '    }\n'
        '    asm volatile("fence.proxy.async.shared::cta;":::"memory");\n'
        '    __syncthreads();\n'
        '  }\n'
        '  uint64_t pw,px;\n'
        '  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pw));\n'
        '  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;":"=l"(px));\n'
        '  auto tma=[&](uint32_t p,const void* tm,int k,int r,uint32_t b,uint64_t policy) {\n'
        '    if constexpr(PIPE&2048)q_tma3(p,tm,k,r,b,policy);else q_tma(p,tm,k,r,b,policy);\n'
        '  };\n'
        '  if constexpr(PIPE&4096)asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pw));\n'
        '  auto load=[&](int i) {\n'
        '    const int stage=i%ST;\n'
        '    q_expect(bars+stage*8,KT*(LR*128+XB));\n'
        '    if constexpr(KT==2)tma(base+stage*SB,&tw,0,(group*128+(kb+i)*KT)*NR,bars+stage*8,pw);\n'
        '    else tma(base+stage*SB,&tw,(kb+i)*64,group*LR,bars+stage*8,pw);\n'
        '#pragma unroll\n'
        '    for(int q=0;q<KT;q++)tma(base+stage*SB+KT*WB+q*XB,&tx,((kb+i)*KT+q)*64,tok,bars+stage*8,px);\n'
        '  };\n'
        '  if(tid==((PIPE&2)?128:0)) {\n'
        '#pragma unroll\n'
        '    for(int i=0;i<ST;i++) {q_init(bars+i*8,1);if constexpr(PIPE&2)q_init(empty+i*8,4);}\n'
        '    if constexpr(S>1) q_init(rb,1);\n'
        '    asm volatile("fence.proxy.async.shared::cta;":::"memory");\n'
        '    if constexpr(S>1 && RED<4) {q_expect(rb,(S-1)*own*16);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n'
        '#pragma unroll\n'
        '    for(int i=0;i<ST;i++) if(i<nkb)load(i);\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if constexpr(S>1 && RED>=4) {\n'
        '    if(tid==((PIPE&2)?128:0)) {\n'
        '      q_expect(rb,(S-1)*own*16);\n'
        '      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");\n'
        '    }\n'
        '    __syncthreads();\n'
        '  }\n'
        '  if constexpr(S>1 && RED!=3) asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");\n'
        '  if constexpr(PIPE&2) {\n'
        '    if(tid>=128) {\n'
        '      if(tid==128) for(int i=ST;i<nkb;i++) {q_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}\n'
        '      if constexpr(S>1) {\n'
        '        if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");\n'
        '        asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n'
        '        if constexpr(RED==1 || RED==3) q_sync();\n'
        '      }\n'
        '      return;\n'
        '    }\n'
        '  }\n'
        '  float a[(TM/64)*(NR/2)];\n'
        '#pragma unroll\n'
        '  for(int h=0;h<TM/64;h++)\n'
        '#pragma unroll\n'
        '    for(int j=0;j<NR/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(a[h*(NR/2)+j]));\n'
        'if constexpr(PIPE&64) asm volatile("wgmma.fence.sync.aligned;":::"memory");\n'
        '#pragma unroll 1\n'
        '  for(int i=0;i<nkb;i++) {\n'
        '    const int stage=i%ST;\n'
        '    q_wait(bars+stage*8,(i/ST)&1);\n'
        '#pragma unroll\n'
        '    for(int h=0;h<TM/64;h++)\n'
        '#pragma unroll\n'
        '      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");\n'
        '    if constexpr(!(PIPE&64)) asm volatile("wgmma.fence.sync.aligned;":::"memory");\n'
        '#pragma unroll\n'
        '    for(int k=0;k<4*KT;k++) {\n'
        '      if constexpr(PIPE&8) {\n'
        '        uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);\n'
        '        @@SWAP@@\n'
        '      } else {\n'
        '#pragma unroll\n'
        '        for(int h=0;h<TM/64;h++) {\n'
        '          uint64_t dx=q_desc(base+stage*SB+KT*WB+(k/4)*XB+h*8192+(k%4)*32),dw=q_desc(base+stage*SB+(k/4)*WB+(k%4)*32);\n'
        '          @@CALL@@\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'
        '    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(((PIPE>>8)&7)?((PIPE>>8)&7):1):"memory");\n'
        '    else asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
        '#pragma unroll\n'
        '    for(int h=0;h<TM/64;h++)\n'
        '#pragma unroll\n'
        '      for(int j=0;j<NR/2;j++) if constexpr(!(PIPE&128)) asm volatile("":"+f"(a[h*(NR/2)+j])::"memory");\n'
        '    int done=i-((PIPE&1)?(((PIPE>>8)&7)?((PIPE>>8)&7):1):0);\n'
        '    if(done>=0 && done+ST<nkb) {\n'
        '      if constexpr(PIPE&2) {if(lane==0)asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(empty+(done%ST)*8):"memory");}\n'
        '      else if(tid==0)load(done+ST);\n'
        '    }\n'
        '  }\n'
        '  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'
        '  if constexpr(PROFILE) stream_end=q_time();\n'
        '  auto emit=[&](int v,float4 z) {\n'
        '    int t=v%128, j=(v/128)%(NR/8), h=(v/128)/(NR/8);\n'
        '    int rr=h*64+(t/32)*16+(t%32)/4, cc=j*8+(t%4)*2;\n'
        '    float vv[4]={z.x,z.y,z.z,z.w};\n'
        '#pragma unroll\n'
        '    for(int c=0;c<4;c++) {\n'
        '      int tr,wc;\n'
        '      if constexpr(PIPE&8) {tr=tok+(v/128)*8+(t%4)*2+(c&1);wc=(t/32)*16+(t%32)/4+(c/2)*8;}\n'
        '      else {tr=tok+rr+(c/2)*8;wc=cc+(c&1);}\n'
        '      if(tr<m && wc<rows)out[tr*2304+row+wc]=__float2bfloat16_rn(vv[c]);\n'
        '    }\n'
        '  };\n'
        '  if constexpr(S==1) {\n'
        '#pragma unroll\n'
        '    for(int h=0;h<TM/64;h++)\n'
        '#pragma unroll\n'
        '      for(int j=0;j<NR/8;j++)emit((h*(NR/8)+j)*128+tid,make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]));\n'
        '  } else if constexpr(RED>=4) {\n'
        '    static_assert(NR==64 && (PIPE&8));\n'
        '    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n'
        '#pragma unroll\n'
        '    for(int j=0;j<TM/8;j++) {\n'
        '      int dst=j%S,local=(j/S)*128+tid,slot=rank-(rank>dst);\n'
        '      if(dst!=rank) q_push(q_map(base+RECV+(slot*OWN+local)*16,dst),\n'
        '        make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]),q_map(rb,dst));\n'
        '    }\n'
        '    q_wait(rb,0);\n'
        '#pragma unroll\n'
        '    for(int j=0;j<TM/8;j++) if(j%S==rank) {\n'
        '      float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '      for(int r=0;r<S;r++) {\n'
        '        float4 z=r==rank?make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]):\n'
        '          *(float4*)(sm+RECV+((r-(r>rank))*OWN+(j/S)*128+tid)*16);\n'
        '        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;\n'
        '      }\n'
        '      if constexpr(RED==4) emit(j*128+tid,sum);\n'
        '      else {\n'
        '        a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;\n'
        '      }\n'
        '    }\n'
        '    if constexpr(RED==5) {\n'
        '      asm volatile("bar.sync 1,128;":::"memory");\n'
        '#pragma unroll\n'
        '      for(int j=0;j<TM/8;j++) if(j%S==rank) {\n'
        '        __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]),hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);\n'
        '        int rr=(tid/32)*16+((tid%16)/8)*8,cc=8*j+tid%8;\n'
        '        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::\n'
        '          "r"(base+2*(cc*72+rr)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");\n'
        '      }\n'
        '      asm volatile("bar.sync 1,128;":::"memory");\n'
        '      for(int q=tid;q<(own/128)*64;q+=128) {\n'
        '        int token=(rank+(q/64)*S)*8+(q/8)%8,rr=(q%8)*8;\n'
        '        if(tok+token<m) *(uint4*)(out+(tok+token)*2304+row+rr)=*(uint4*)((__nv_bfloat16*)sm+token*72+rr);\n'
        '      }\n'
        '    }\n'
        '  } else {\n'
        '    if constexpr(RED==3) asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");\n'
        '    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");\n'
        '#pragma unroll\n'
        '    for(int h=0;h<TM/64;h++)\n'
        '#pragma unroll\n'
        '      for(int j=0;j<NR/8;j++) {\n'
        '        int v=(h*(NR/8)+j)*128+tid,dst=v/OWN,local=v%OWN;\n'
        '        float4 z=make_float4(a[h*(NR/2)+j*4],a[h*(NR/2)+j*4+1],a[h*(NR/2)+j*4+2],a[h*(NR/2)+j*4+3]);\n'
        '        uint32_t p=base+RECV+(rank*OWN+local)*16;\n'
        '        if(dst==rank)*(float4*)(sm+RECV+(rank*OWN+local)*16)=z;\n'
        '        else q_push(q_map(p,dst),z,q_map(rb,dst));\n'
        '      }\n'
        '    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;":::"memory");else __syncthreads();\n'
        '    q_wait(rb,0);\n'
        '    for(int t=tid;t<own;t+=128) {\n'
        '      float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '      for(int r=0;r<S;r++) {\n'
        '        float4 z=*(float4*)(sm+RECV+(r*OWN+t)*16);\n'
        '        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;\n'
        '      }\n'
        '      emit(rank*OWN+t,sum);\n'
        '    }\n'
        '    if constexpr(RED==1 || RED==3) q_sync();\n'
        '  }\n'
        '  if constexpr(PROFILE) if(lane==0) {\n'
        '    unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));\n'
        '    int idx=((blockIdx.y*gridDim.x+blockIdx.x)*4+tid/32)*4;\n'
        '    stamps[idx]=smid;stamps[idx+1]=begin;stamps[idx+2]=stream_end;stamps[idx+3]=q_time();\n'
        '  }\n'
        '}\n'
        'using QEncode=decltype(&cuTensorMapEncodeTiled);\n'
        'int q_tensor_map(CUtensorMap* tm,int64_t p,int inner,int rows,int box,bool map3) {\n'
        '  static QEncode encode=nullptr;\n'
        '  if(!encode){cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);if(!encode)return -100;}\n'
        '  if(map3) {\n'
        '    uint64_t dims[3]={64,(uint64_t)inner/64,(uint64_t)rows},stride[2]={128,(uint64_t)inner*2};\n'
        '    uint32_t box3[3]={64,1,(uint32_t)box},es[3]={1,1,1};\n'
        '    return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)p,dims,stride,box3,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n'
        '      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n'
        '  }\n'
        '  uint64_t dims[2]={(uint64_t)inner,(uint64_t)rows},stride[1]={(uint64_t)inner*2};uint32_t b[2]={64,(uint32_t)box},es[2]={1,1};\n'
        '  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,\n'
        '    CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n'
        '}\n'
        'template<int NR,int G,int TM,int S,int ST,int PIPE,int RED,bool PROFILE>\n'
        'int qb_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {\n'
        '  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);\n'
        '  constexpr int KT=(PIPE&32)?2:1;\n'
        '  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*16:0);\n'
        '  auto fn=qb_kernel<NR,G,TM,S,ST,PIPE,RED,PROFILE>;\n'
        '  static bool ready=false;\n'
        '  if(!ready){cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e!=cudaSuccess)return -(int)e;\n'
        '    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);if(e!=cudaSuccess)return -(int)e;ready=true;}\n'
        '  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n'
        '  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(G*S,(m+TM-1)/TM);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n'
        '  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={S,1,1};if(S>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n'
        '  if(query==1){int v=0;cudaError_t e=S>1?cudaOccupancyMaxActiveClusters(&v,fn,&cfg):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,fn,cfg.blockDim.x,bytes);return e==cudaSuccess?v:-(int)e;}\n'
        '  CUtensorMap tw,tx;int e=q_tensor_map(&tw,w,KT==2?64:8192,KT==2?G*128*NR:G*LR,KT==2?KT*NR:LR,PIPE&2048);if(e)return e;e=q_tensor_map(&tx,x,8192,m,TM,PIPE&2048);if(e)return e;\n'
        '  return int(cudaLaunchKernelEx(&cfg,fn,tw,tx,(__nv_bfloat16*)out,m,(unsigned long long*)stats));\n'
        '}\n'
        'int qb_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats) {\n'
        '  switch(id){@@CASES@@ default:return -99;}\n'
        '}\n'
    )

    def tt_v14_qkv_build_qkv(configs, profile=False):
        ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
        source = tt_v14_qkv_QKVBIG_CUDA.replace('@@MMA@@', '\n'.join((tt_v14_qkv_qbig_mma(n) for n in ns)))
        source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
        source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
        source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return qb_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs))))
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        namespace = 'qb_' + digest
        begin = source.index('#define QD')
        args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
        source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
        cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
        return load_inline('qbig_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

    def tt_v14_qkv_qbig_pack(w, groups, kt=1):
        rows = (2304 + groups - 1) // groups
        if kt == 2:
            return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
        if 2304 % groups == 0:
            return w.contiguous()
        out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
        for g in range(groups):
            lo, hi = (g * 2304 // groups, (g + 1) * 2304 // groups)
            out[g, :hi - lo].copy_(w[lo:hi])
        return out.reshape(-1, 8192)

    def tt_v14_qkv_prepare_qbig_candidates(weights, configs, profile=False):
        w = weights['qkv_gate']
        return {'ext': tt_v14_qkv_build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): tt_v14_qkv_qbig_pack(w, g, kt) for g, kt in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

    def tt_v14_qkv_run_qbig_candidate(x, st, index, stats=None, query=0):
        out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
        cfg = st['configs'][index]
        rc = st['ext'].qb_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
        if query:
            return rc
        if rc:
            raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
        return out

    tt_v14_qkv_EXPERIMENT_CONFIGS = [(64, 36, 128, 3, 8, 539, 5), (64, 36, 128, 3, 8, 6683, 5), (64, 36, 256, 3, 4, 219, 5)]

    tt_v14_qkv_ENABLED_ROUTES = {128: 0}

    def tt_v14_qkv_prepare_qkvbig(weights: dict) -> dict:
        w = weights.get('qkv_gate')
        if w is None or w.shape != (2304, 8192) or w.dtype != torch.bfloat16 or (not w.is_cuda):
            return {'routes': {}}
        if torch.cuda.get_device_capability(w.device) != (9, 0) or 'H200' not in torch.cuda.get_device_name(w.device):
            return {'routes': {}}
        if not tt_v14_qkv_ENABLED_ROUTES:
            return {'routes': {}}
        state = tt_v14_qkv_prepare_qbig_candidates(weights, tt_v14_qkv_EXPERIMENT_CONFIGS)
        state['routes'] = dict(tt_v14_qkv_ENABLED_ROUTES)
        for m, index in state['routes'].items():
            x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
            tt_v14_qkv_run_qbig_candidate(x, state, index)
        torch.cuda.synchronize(w.device)
        return state

    def tt_v14_qkv_run_qkvbig(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
        if layer != 'qkv_gate' or n != 2304 or x.ndim != 2 or (x.shape[1] != 8192):
            return None
        index = st.get('routes', {}).get(x.shape[0])
        if index is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
            return None
        if x.device != st['device'] or x.data_ptr() % 16:
            return None
        return tt_v14_qkv_run_qbig_candidate(x, st, index)

    dd_decodediet_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}

    dd_decodediet_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

    def dd_decodediet_dd_cap2_b_cps32_mt(m):
        return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

    dd_decodediet_CPSSM_CUDA_TEMPLATE = (
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#include <type_traits>\n'
        '#include <cooperative_groups.h>\n'
        '\n'
        '#define CPSSD __device__ __forceinline__\n'
        '\n'
        'CPSSD int4 dd_cap2_b_cpssm_ldw(const int4* p) {\n'
        '  int4 v;\n'
        '  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n'
        '               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'CPSSD uint4 dd_cap2_b_cpssm_ldx(const void* p) {\n'
        '  uint4 v;\n'
        '  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'CPSSD uint32_t dd_cap2_b_cpssm_lde(const uint32_t* p) {\n'
        '  uint32_t v;\n'
        '  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'CPSSD uint32_t dd_cap2_b_cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {\n'
        '  uint32_t d;\n'
        '  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n'
        '  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n'
        '      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n'
        '}\n'
        '\n'
        '// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\n'
        'CPSSD void dd_cap2_b_cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n'
        '  uint32_t rotated;\n'
        '  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n'
        '  const uint32_t low_high = (C & 0x87878787u) + T0;\n'
        '  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n'
        '  a[0] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x5140u);\n'
        '  a[1] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x7362u);\n'
        '  a[2] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x5140u);\n'
        '  a[3] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x7362u);\n'
        '}\n'
        '\n'
        '__device__ __noinline__ void dd_cap2_b_cpssm_never(void* p) { asm volatile("trap;"); }\n'
        '\n'
        '// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\n'
        'CPSSD void dd_cap2_b_cpssm_order_point(int m, void* p) {\n'
        '  uint32_t t;\n'
        '  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n'
        '  if ((t & (uint32_t)m) == 0x7fffffffu) dd_cap2_b_cpssm_never(p);\n'
        '}\n'
        '\n'
        'CPSSD uint32_t dd_cap2_b_cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        '\n'
        'CPSSD uint32_t dd_cap2_b_cpssm_rank() {\n'
        '  uint32_t r;\n'
        '  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n'
        '  return r;\n'
        '}\n'
        '\n'
        'CPSSD uint32_t dd_cap2_b_cpssm_mapa(uint32_t a, uint32_t rank) {\n'
        '  uint32_t d;\n'
        '  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_mbar_init(uint32_t bar, uint32_t count) {\n'
        '  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_mbar_expect(uint32_t bar, uint32_t tx) {\n'
        '  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\n'
        'CPSSD void dd_cap2_b_cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\n'
        'CPSSD void dd_cap2_b_cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n'
        '               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n'
        '}\n'
        '\n'
        'CPSSD bool dd_cap2_b_cpssm_mbar_try(uint32_t bar) {\n'
        '  uint32_t ok;\n'
        '  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n'
        '               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n'
        '  return ok != 0;\n'
        '}\n'
        '\n'
        'CPSSD uint64_t dd_cap2_b_cpssm_gtime() {\n'
        '  uint64_t t;\n'
        '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n'
        '  return t;\n'
        '}\n'
        '\n'
        '// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.\n'
        'CPSSD void dd_cap2_b_cpssm_wait_bar(uint32_t bar) {\n'
        '  if (dd_cap2_b_cpssm_mbar_try(bar)) return;\n'
        '  const uint64_t t0 = dd_cap2_b_cpssm_gtime();\n'
        '  while (!dd_cap2_b_cpssm_mbar_try(bar))\n'
        '    if (dd_cap2_b_cpssm_gtime() - t0 > 20000000ull) __trap();\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n'
        '\n'
        'CPSSD float4 dd_cap2_b_cpssm_shfl_xor4(const float4& v, int o) {\n'
        '  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n'
        '                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n'
        '}\n'
        '\n'
        'CPSSD void dd_cap2_b_cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\n'
        'CPSSD void dd_cap2_b_cpssm_store(float* p, float v) { *p = v; }\n'
        '\n'
        'template <int FMT> struct CpsSmFmt;\n'
        'template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };\n'
        'template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };\n'
        'template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };\n'
        'template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };\n'
        '\n'
        'template <int FMT>\n'
        'CPSSD void dd_cap2_b_cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n'
        '                   uint32_t U0, uint32_t U1) {\n'
        '  if constexpr (FMT == 0 || FMT == 2) {\n'
        '#pragma unroll\n'
        '    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {\n'
        '      A[4 * l + 0] = (uint32_t)w[l].x;\n'
        '      A[4 * l + 1] = (uint32_t)w[l].y;\n'
        '      A[4 * l + 2] = (uint32_t)w[l].z;\n'
        '      A[4 * l + 3] = (uint32_t)w[l].w;\n'
        '    }\n'
        '  } else {\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {\n'
        '      const int4* u = w + 3 * h;\n'
        '      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n'
        '      dd_cap2_b_cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n'
        '      dd_cap2_b_cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n'
        '      dd_cap2_b_cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n'
        '      dd_cap2_b_cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        "// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.\n"
        'template <int NR>\n'
        'CPSSD void dd_cap2_b_cpssm_patch(uint32_t (&A)[NR], uint32_t e) {\n'
        '  const uint32_t reg = (e >> 8) & 31u;\n'
        '  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n'
        '#pragma unroll\n'
        '  for (int r = 0; r < NR; ++r)\n'
        '    if (reg == (uint32_t)r) A[r] ^= xr;\n'
        '}\n'
        '\n'
        "// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.\n"
        'template <class Emit>\n'
        'CPSSD void dd_cap2_b_cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n'
        '  for (int idx = tid; idx < NI; idx += nt) {\n'
        '    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n'
        '    for (int p = 0; p < P; ++p) dd_cap2_b_cpssm_add4(v, src[(size_t)p * PSTR + idx]);\n'
        '    emit(idx, v);\n'
        '  }\n'
        '}\n'
        '\n'
        '// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n'
        '// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n'
        '// straight-line code block at the end, 128 (with 4) no output stores.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>\n'
        '__global__ void __launch_bounds__(NTMAX) dd_cap2_b_cpssm_gemv(\n'
        '    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n'
        '    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n'
        '    uint32_t U0, uint32_t U1) {\n'
        '  const int m = CM ? CM : m_arg;\n'
        '  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n'
        '  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n'
        '  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;\n'
        '  constexpr bool C12 = FMT == 1 || FMT == 3;\n'
        '  constexpr bool HILO = FMT >= 2;\n'
        '  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n'
        '  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n'
        '  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n'
        '  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n'
        '  OutT* out = reinterpret_cast<OutT*>(outp);\n'
        '  extern __shared__ __align__(16) unsigned char smem[];\n'
        '  __shared__ __align__(8) uint64_t s_bar;\n'
        '  const int rawbytes = CP * WR * WK * RT * NL * 512;\n'
        '  if constexpr (EMPTY) {\n'
        '    return;\n'
        '  } else {\n'
        '    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n'
        '    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n'
        '    const int S = CS ? CS : gridDim.y;\n'
        '    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n'
        '    const int G = GX * S * NW;\n'
        '    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n'
        '    const int rt0 = (blockIdx.x * WR + wr) * RT;\n'
        '    const int c0 = (blockIdx.y * WK + wk) * KU;\n'
        '    const int TV = min(4 * MT, (m + 1) >> 1);\n'
        '    const uint32_t bar = dd_cap2_b_cpssm_smem_u32(&s_bar);\n'
        '    uint32_t rank = 0;\n'
        '    if constexpr (PUSH) {\n'
        '      rank = dd_cap2_b_cpssm_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        dd_cap2_b_cpssm_mbar_init(bar, 1);\n'
        '        dd_cap2_b_cpssm_mbar_expect(bar, parts * RT * TV * 128u);\n'
        '        dd_cap2_b_cpssm_fence_init();\n'
        '      }\n'
        '      dd_cap2_b_cpssm_cl_arrive();\n'
        '    }\n'
        '\n'
        '    uint32_t my_e = 0xFFFFFFFFu;\n'
        '    if constexpr (C12 && !RO) my_e = dd_cap2_b_cpssm_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n'
        '    const size_t sstride = (size_t)G * (RT * NL * 32);\n'
        '    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n'
        '\n'
        '    int4* rawring = reinterpret_cast<int4*>(smem);\n'
        '    auto stage_weights = [&](int slot, int si) {\n'
        '      if constexpr (CP > 0) {\n'
        '        if (si < KU) {\n'
        '#pragma unroll\n'
        '          for (int j=0; j<RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int l=0; l<NL; ++l) {\n'
        '              const uint32_t dst=dd_cap2_b_cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);\n'
        '              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;\n'
        '              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");\n'
        '            }\n'
        '        }\n'
        '        asm volatile("cp.async.commit_group;" ::: "memory");\n'
        '      }\n'
        '    };\n'
        '#pragma unroll\n'
        '    for (int q=0; q<CP; ++q) stage_weights(q,q);\n'
        '    int4 w[ST][RT][NL];\n'
        '    uint4 xv[ST][MT][2];\n'
        '    auto load = [&](int slot, int s) {\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = dd_cap2_b_cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n'
        '      if constexpr (!NOX && !RO) {\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n) {\n'
        '          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n'
        '          xv[slot][n][0] = dd_cap2_b_cpssm_ldx(xp);\n'
        '          xv[slot][n][1] = dd_cap2_b_cpssm_ldx(xp + 8);\n'
        '        }\n'
        '      }\n'
        '    };\n'
        '#pragma unroll\n'
        '    for (int s = 0; s < ST; ++s) load(s, s);\n'
        '    dd_cap2_b_cpssm_order_point(m, outp);\n'
        '\n'
        '    float acc[NA][RT][MT][4];\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < NA; ++h)\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n'
        '    uint32_t fold = 0;\n'
        '    uint32_t nxt = 0xFFFFFFFFu;\n'
        '    int ep = 0;\n'
        '    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n'
        '\n'
        '#pragma unroll 1\n'
        '    for (int i = 0; i < KU; i += ST) {\n'
        '#pragma unroll\n'
        '      for (int s = 0; s < ST; ++s) {\n'
        '        const int si = i + s;\n'
        '        if (ST == 3 && si >= KU) break;\n'
        '        if constexpr (RO) {\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int l = 0; l < NL; ++l)\n'
        '              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          dd_cap2_b_cpssm_order_point(m, outp);\n'
        '        } else {\n'
        '          if constexpr (CP > 0) {\n'
        '            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");\n'
        '#pragma unroll\n'
        '            for (int j=0; j<RT; ++j)\n'
        '#pragma unroll\n'
        '              for (int l=0; l<NL; ++l) {\n'
        '                const uint32_t src=dd_cap2_b_cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);\n'
        '                int4& v=w[s][j][l];\n'
        '                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");\n'
        '              }\n'
        '          }\n'
        '          uint32_t A[RT][NA * 16];\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j) dd_cap2_b_cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n'
        '          if constexpr (CP > 0) stage_weights(si%CP,si+CP);\n'
        '          uint32_t b[MT][8];\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n) {\n'
        '            if constexpr (NOX) {\n'
        '#pragma unroll\n'
        '              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n'
        '            } else {\n'
        '              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n'
        '              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n'
        '            }\n'
        '          }\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          dd_cap2_b_cpssm_order_point(m, outp);\n'
        '          if constexpr (C12) {\n'
        '#pragma unroll\n'
        '            for (int j = 0; j < RT; ++j) {\n'
        '              const uint32_t useq = (uint32_t)(si * RT + j);\n'
        '              if ((nxt >> 18) == useq) {\n'
        '                do {\n'
        '                  if (lane == (int)((nxt >> 13) & 31u)) dd_cap2_b_cpssm_patch<NA * 16>(A[j], nxt);\n'
        '                  ++ep;\n'
        '                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n'
        '                } while ((nxt >> 18) == useq);\n'
        '              }\n'
        '            }\n'
        '          }\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '              for (int q = 0; q < 4; ++q) {\n'
        '                if constexpr (NOMMA) {\n'
        '#pragma unroll\n'
        '                  for (int h = 0; h < NA; ++h)\n'
        '                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n'
        '                            A[j][16 * h + 4 * q + 3];\n'
        '                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n'
        '                } else {\n'
        '                  dd_cap2_b_cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                  if constexpr (HILO) dd_cap2_b_cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                }\n'
        '              }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '\n'
        '    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");\n'
        '    if constexpr (RO) {\n'
        '      if constexpr (RO_CODE) {\n'
        '        if (warp == 0) {\n'
        '          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n'
        '          const uint32_t k = (uint32_t)m | 1u;\n'
        '#pragma unroll\n'
        '          for (int q = 0; q < 64; ++q)\n'
        '            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n'
        '                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n'
        '          fold = a ^ b ^ c ^ d;\n'
        '        }\n'
        '      }\n'
        '      if constexpr (RO_STORE) {\n'
        '        if ((lane & 3) == 0)\n'
        '          dd_cap2_b_cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n'
        '      } else {\n'
        '        if (fold == 0x9e3779b9u) dd_cap2_b_cpssm_store(out, 1.f);\n'
        '      }\n'
        '      return;\n'
        '    } else {\n'
        '      float r[RT][MT][4];\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n'
        '      if constexpr (NOMMA) {\n'
        '        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n'
        '      }\n'
        '\n'
        '      const int RB = WR * RT * 16;\n'
        '      const int row0 = blockIdx.x * RB;\n'
        '      if (NORED || (RED == 0 && WK == 1)) {\n'
        '#pragma unroll\n'
        '        for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '            for (int c = 0; c < 4; ++c) {\n'
        '              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n'
        '              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n'
        '              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n'
        '                dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);\n'
        '            }\n'
        '        return;\n'
        '      }\n'
        '      if constexpr (!NORED) {\n'
        '        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n'
        '        auto emit_full = [&](int idx, float4 v) {\n'
        '          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n'
        '            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n'
        '            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        auto emit_compact = [&](int idx, float4 v) {\n'
        '          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = 2 * t + (c & 1);\n'
        '            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n'
        '            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        if constexpr (RED == 0 || RED == 1) {\n'
        '          float4* part = reinterpret_cast<float4*>(smem + rawbytes);\n'
        '          const int NI = WR * RT * MT * 32;\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n'
        '                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '          __syncthreads();\n'
        '          if constexpr (RED == 0) {\n'
        '            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);\n'
        '          } else {\n'
        '            float4* csum = part + (size_t)NW * RT * MT * 32;\n'
        '            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n'
        '            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n'
        '            cl.sync();\n'
        '            const int rk = (int)cl.block_rank();\n'
        '            const int per = (NI + S - 1) / S;\n'
        '            const int e1 = min(NI, (rk + 1) * per);\n'
        '            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n'
        '              float4 v = *cl.map_shared_rank(csum + idx, 0);\n'
        '              for (int q = 1; q < S; ++q) dd_cap2_b_cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));\n'
        '              emit_full(idx, v);\n'
        '            }\n'
        '            cl.sync();\n'
        '          }\n'
        '        } else {\n'
        '          const int NIc = WR * RT * TV * 8;\n'
        '          const int t = lane & 3, g = lane >> 2;\n'
        '          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);\n'
        "          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.\n"
        '          if constexpr (RED == 2) {\n'
        '            dd_cap2_b_cpssm_cl_wait();\n'
        '            if (t < TV) {\n'
        '              const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n) {\n'
        '                  const int tn = n * 4 + t;\n'
        '                  if (tn < TV) {\n'
        '                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;\n'
        '                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n'
        '                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '                    if (rank == 0) *dst = v;\n'
        '                    else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);\n'
        '                  }\n'
        '                }\n'
        '            }\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            dd_cap2_b_cpssm_wait_bar(bar);\n'
        '            dd_cap2_b_cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n'
        '          } else {\n'
        '            float4* part = recv + (size_t)S * NIc;\n'
        '            if (t < TV) {\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n)\n'
        '                  if (n * 4 + t < TV)\n'
        '                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =\n'
        '                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '            }\n'
        '            __syncthreads();\n'
        '            dd_cap2_b_cpssm_cl_wait();\n'
        '            const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);\n'
        '            dd_cap2_b_cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n'
        '              float4* dst = recv + (size_t)rank * NIc + idx;\n'
        '              if (rank == 0) *dst = v;\n'
        '              else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);\n'
        '            });\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            dd_cap2_b_cpssm_wait_bar(bar);\n'
        '            dd_cap2_b_cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'static int dd_cap2_b_cpssm_smem_optin() {\n'
        '  static int v = -1;\n'
        '  if (v < 0) {\n'
        '    int dev = 0;\n'
        '    cudaGetDevice(&dev);\n'
        '    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n'
        '  }\n'
        '  return v;\n'
        '}\n'
        '\n'
        '// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>\n'
        'static int dd_cap2_b_cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  auto kern = dd_cap2_b_cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;\n'
        '  const int policy = query / 100;\n'
        '  query %= 100;\n'
        '  if (query == 2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n'
        '    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n'
        '  }\n'
        '  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n'
        '  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n'
        '  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n'
        '  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n'
        '  if (N % RB) return -2;\n'
        '  if ((int64_t)S * WK * KU * 64 != K) return -3;\n'
        '  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n'
        '  if (m < 1 || m > MT * 8) return -5;\n'
        '  const bool cl = RED != 0;\n'
        '  if (!cl && S != 1) return -9;\n'
        '  if (cl && (S < 2 || S > 16)) return -9;\n'
        '  const int TV = std::min(4 * MT, (m + 1) / 2);\n'
        '  size_t sm = 0;\n'
        '  if (!(DIAG & 28)) {\n'
        '    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n'
        '    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n'
        '    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n'
        '    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n'
        '  }\n'
        '  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;\n'
        '  static int max_dyn = -1;\n'
        '  if (max_dyn < 0) {\n'
        '    cudaFuncAttributes fa;\n'
        '    cudaFuncGetAttributes(&fa, kern);\n'
        '    max_dyn = dd_cap2_b_cpssm_smem_optin() - (int)fa.sharedSizeBytes;\n'
        '    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n'
        '    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n'
        '  }\n'
        '  if ((int64_t)sm > max_dyn) return -6;\n'
        '  cudaLaunchConfig_t lc = {};\n'
        '  cudaLaunchAttribute at[2];\n'
        '  lc.gridDim = dim3(N / RB, S, 1);\n'
        '  lc.blockDim = dim3(nt, 1, 1);\n'
        '  lc.dynamicSmemBytes = sm;\n'
        '  lc.stream = (cudaStream_t)stream;\n'
        '  if (cl) {\n'
        '    at[0].id = cudaLaunchAttributeClusterDimension;\n'
        '    at[0].val.clusterDim.x = 1;\n'
        '    at[0].val.clusterDim.y = S;\n'
        '    at[0].val.clusterDim.z = 1;\n'
        '    lc.attrs = at;\n'
        '    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;\n'
        '    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);\n'
        '    lc.numAttrs = 2;\n'
        '  }\n'
        '  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }\n'
        '  if (query == 4) return (int)sm;\n'
        '  if (query == 1) {\n'
        '    int v = 0;\n'
        '    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n'
        '                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n'
        '    cudaGetLastError();\n'
        '    return e == cudaSuccess ? v : -11;\n'
        '  }\n'
        '  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n'
        '                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n'
        '                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n'
        '                     (uint32_t)((uint64_t)u01 >> 32));\n'
        '  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n'
        '}\n'
        '\n'
        'int dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  switch (inst) {\n'
        '@@CASES@@\n'
        '    default: return -99;\n'
        '  }\n'
        '}\n'
    )

    dd_decodediet_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

    dd_decodediet_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}

    dd_decodediet_CPSSM_ESC_CAP = 32

    def dd_decodediet_dd_cap2_b_cpssm_units(w):
        """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
        return v.reshape(n // 16, k // 64, 32, 32)

    def dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v):
        r, c = v.shape[:2]
        words = dd_decodediet_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
        return words.view(torch.uint8).view(r, c, 2048)

    def dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):
        """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
        r, c = v.shape[:2]
        dev = v.device
        lo = v & 255
        sign = v >> 15
        eh = v >> 8 & 127
        hist = torch.bincount(eh.flatten(), minlength=128).cpu()
        window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
        base = int(window.argmax())
        d = eh - base
        inwin = (d >= 0) & (d < 8)
        d = torch.where(inwin, d, torch.zeros_like(d))
        nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
        pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
        code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
        code = dd_decodediet_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
        lo8 = lo.to(torch.uint8)
        data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
        er, ec, el, ev = (~inwin).nonzero().unbind(1)
        xor7 = eh[er, ec, el, ev] ^ base
        return (data, base, (er, ec, el, ev, xor7))

    def dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
        """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
        r, c = data.shape[:2]
        dev = data.device
        p = data.view(r, c, 3, 32, 16)
        lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
        code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
        nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
        pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
        d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
        sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
        val = (sg << 15 | base + d << 8).to(torch.int32) | lo
        er, ec, el, ev, xor7 = esc
        val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
        return val

    def dd_decodediet_dd_cap2_b_cpssm_table(base):
        return base * 16843009

    def dd_decodediet_dd_cap2_b_cpssm_pack_format(w, fmt):
        """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

        Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
        half 1 marks the lo matrix of a HILO pair."""
        if fmt in (2, 3):
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            mats = [dd_decodediet_dd_cap2_b_cpssm_units(hi), dd_decodediet_dd_cap2_b_cpssm_units(lo)]
        else:
            mats = [dd_decodediet_dd_cap2_b_cpssm_units(w)]
        if fmt in (0, 2):
            data = torch.cat([dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
            return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
        parts, tabs, escs = ([], [], [])
        for h, v in enumerate(mats):
            data, base, esc = dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v)
            if not torch.equal(dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
                return None
            parts.append(data)
            tabs.append(dd_decodediet_dd_cap2_b_cpssm_table(base))
            escs.append(esc + (torch.full_like(esc[0], h),))
        esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
        return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

    def dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
        if R % (wr * rt) or C != s * wk * ku:
            return None
        gx = R // (wr * rt)
        return (gx, wr * wk, gx * s * wr * wk)

    def dd_decodediet_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
        """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
        data = pk['data']
        R, C, B = data.shape
        geo = dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
        if geo is None:
            return None
        gx, nw, G = geo
        dev = data.device
        si = torch.arange(ku, device=dev).view(ku, 1, 1)
        gw = torch.arange(G, device=dev).view(1, G, 1)
        j = torch.arange(rt, device=dev).view(1, 1, rt)
        cta, warp = (gw // nw, gw % nw)
        by, bx = (cta // gx, cta % gx)
        wri, wki = (warp % wr, warp // wr)
        tile = (bx * wr + wri) * rt + j
        chunk = (by * wk + wki) * ku + si
        idx = (tile * C + chunk).reshape(-1)
        wp = data.view(R * C, B)[idx].contiguous().view(-1)
        esc = torch.full((G, dd_decodediet_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
        if pk['esc'] is not None and pk['esc'][0].numel():
            er, ec, el, ev, x7, half = pk['esc']
            bx_e, rem = (er // (wr * rt), er % (wr * rt))
            wr_e, j_e = (rem // rt, rem % rt)
            by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
            wk_e, s_e = (rem2 // ku, rem2 % ku)
            g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
            seq = s_e * rt + j_e
            if int(seq.max()) >= (1 << 14) - 1:
                return None
            counts = torch.bincount(g_e, minlength=G)
            if int(counts.max()) > dd_decodediet_CPSSM_ESC_CAP:
                return None
            order = torch.argsort(g_e * (1 << 14) + seq)
            g_s, seq_s = (g_e[order], seq[order])
            rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
            reg = ev[order] // 2 + 16 * half[order]
            entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
            esc[g_s, rank] = dd_decodediet_cpstage___to_i32(entry)
        return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

    def dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
        """cfg = (WR, WK, KU, S)."""
        wr, wk, ku, s = cfg
        m, k = x.shape
        return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

    dd_decodediet_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}

    dd_decodediet_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)), ('pooler', 2): ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 2): ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4)), ('unpooler', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))}

    def dd_decodediet_dd_cap2_b_cpssm_rows(layer, m, n_full):
        return 1024 if layer == 'unpooler' and m <= 8 else n_full

    def dd_decodediet_dd_cap2_b_cpssm_bf16_twin(inst):
        """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
        return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

    def dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
        """Runs every column as a one-hot input: each output must equal its weight column exactly.

        exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
        n, k = exact.shape
        mm = rows_per_launch
        dev = exact.device
        out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
        got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
        eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
        for i in range(k // mm):
            if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
                return False
        return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

    def dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
        """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
        n, k = w_rows.shape
        is_router = w_rows.dtype == torch.float32
        tol = 0.0001 if is_router else 0.01
        gen = torch.Generator(device=w_rows.device).manual_seed(seed)
        for m in ms:
            x = torch.randn((m, k), generator=gen, device=w_rows.device)
            x[m - 1].zero_()
            x[m - 1, (seed * 131 + m) % k] = 16.0
            x = x.to(torch.bfloat16)
            ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
            out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
            if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
                return False
            diff = out.double() - ref
            rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
            if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
                return False
        return True

    def dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt):
        if fmt in (2, 3):
            hi = w_rows.to(torch.bfloat16)
            return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
        return w_rows

    def dd_decodediet_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
        """Packs, stream-orders and verifies one table entry; None if any step fails."""
        m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
        rows = dd_decodediet_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
        fmt = inst[0]
        pkey = (layer, rows, fmt)
        if pkey not in packs:
            packs[pkey] = dd_decodediet_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
        if packs[pkey] is None:
            return None
        skey = pkey + (inst[2],) + tuple(cfg)
        if skey not in streams:
            streams[skey] = dd_decodediet_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
        sp = streams[skey]
        if sp is None:
            return None
        vkey = ('verified', inst) + skey
        if vkey not in streams:
            w_rows = w[:rows].contiguous()
            lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
            fixed_m = inst[7] if len(inst) > 7 else 0
            ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
            streams[vkey] = dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
        if not streams[vkey]:
            return None
        return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

    def dd_decodediet_cpstage___to_i16(v):
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

    def dd_decodediet_cpstage___to_i32(v):
        return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

    dd_decodediet_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

    def dd_decodediet_cps_build(family, instances, depth):
        is32 = False
        prefix = 'dd_cap2_b_cpssm_'
        template = dd_decodediet_CPSSM_CUDA_TEMPLATE
        cpp = dd_decodediet_CPSSM_CPP
        cases = []
        for i, inst in enumerate(instances):
            full = inst if len(inst) == 14 else inst + (0,) * 7
            args = ', '.join(map(str, (*full, depth)))
            cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
        source = template.replace('@@CASES@@', '\n'.join(cases))
        name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
        return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

    def dd_decodediet_prepare_cps(weights, depth=None, layers=dd_decodediet_CPS_LAYERS, buckets=(2, 4), include_policy=False):
        if depth is None:
            st = {'plans': {}, 'depth': 'selected', 'policy': 0}
            selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
            for bucket, selected_depth in ((2, 2), (4, 1)):
                if bucket in buckets and selected_layers:
                    part = dd_decodediet_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                    st['plans'].update(part['plans'])
            return st
        if depth not in (0, 1, 2, 3):
            raise ValueError('ring depth must be 0, 1, 2 or 3')
        st = {'plans': {}, 'depth': depth, 'policy': 0}
        for family in ('sm',):
            table = dd_decodediet_CPSSM_TABLE_MT
            entries = []
            for (layer, bucket), (inst, cfg) in table.items():
                is_policy = include_policy and family == 'sm' and (layer in dd_decodediet_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
                if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                    continue
                if not is_policy and family == 'sm' and ((layer, bucket) in dd_decodediet_CPS32_TABLE_MT):
                    continue
                if len(inst) == 7:
                    rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                    inst = inst + (0, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'mt', bucket), (inst, cfg)))
            if include_policy and family == 'sm':
                for (layer, m), (inst, cfg) in dd_decodediet_CPSSM_TABLE_M.items():
                    if layer not in dd_decodediet_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                        continue
                    rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                    if len(inst) == 7:
                        inst = inst + (m, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
                    entries.append(((layer, 'm', m), (inst, cfg)))
            if not entries:
                continue
            twin = dd_decodediet_dd_cap2_b_cpssm_bf16_twin
            instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
            ext = dd_decodediet_cps_build(family, instances, depth)
            ids = {inst: i for i, inst in enumerate(instances)}
            plan_fn = dd_decodediet_dd_cap2_b_cpssm_plan
            packs, streams = ({}, {})
            for key, (inst, cfg) in entries:
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
                chosen = inst
                if sp is None and twin(inst) is not None:
                    chosen = twin(inst)
                    sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
                if sp is not None:
                    st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
        return st

    def dd_decodediet_cps_launch(sp, x, out, n, policy=0, query=0):
        fn = dd_decodediet_dd_cap2_b_cpssm_launch_raw
        return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

    def dd_decodediet_run_cps(x, layer, n, st):
        if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
            return None
        m = x.shape[0]
        sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', dd_decodediet_dd_cap2_b_cps32_mt(m)))
        if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != dd_decodediet_CPS32_LAYER_K[layer]) or (not x.is_cuda):
            return None
        if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
            return None
        out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
        rc = dd_decodediet_cps_launch(sp, x, out, n, st.get('policy', 0))
        if rc:
            raise RuntimeError(f'cpstage launch failed: {rc}')
        return out

    def dd_prepare_decodediet(weights) -> dict:
        routes = dd_DECODEDIET_ROUTES
        state = {'routes': frozenset(routes)}
        if not routes:
            return state
        if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
            layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
            state['cps'] = dd_decodediet_prepare_cps(weights, layers=layers, buckets=(2,))
        return state

    def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
        m = x.shape[0] if x.ndim == 2 else 0
        if (layer, m) not in state.get('routes', ()):
            return None
        return dd_decodediet_run_cps(x, layer, n, state['cps'])

    rt__CUDA = (
        '\n'
        '#include <cuda.h>\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#define RD __device__ __forceinline__\n'
        'RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        'RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }\n'
        'RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }\n'
        'RD void routertail_wait(uint32_t b, int ph) {\n'
        '  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");\n'
        '}\n'
        'RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }\n'
        'RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }\n'
        'RD void routertail_push(uint32_t d, float4 v, uint32_t b) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");\n'
        '}\n'
        'RD float4 routertail_pull(uint32_t d) {\n'
        '  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;\n'
        '}\n'
        'RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }\n'
        'RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }\n'
        'RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {\n'
        '  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");\n'
        '}\n'
        '\n'
        'template<int MT,int S,int WK>\n'
        '__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {\n'
        '  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;\n'
        '  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;\n'
        '  const int gw=(rank*16+row/16)*WK+warp;\n'
        '  extern __shared__ __align__(16) unsigned char sm[];\n'
        '  float4* recv=(float4*)sm;\n'
        '  float4* part=recv+PARTS*OWN;\n'
        '  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);\n'
        '  int4 w[KU][8];uint4 xx[KU][MT][2];\n'
        '#pragma unroll\n'
        '  for(int k=0;k<KU;k++) {\n'
        '#pragma unroll\n'
        '    for(int j=0;j<8;j++) {\n'
        '      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;\n'
        '      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));\n'
        '    }\n'
        '#pragma unroll\n'
        '    for(int n=0;n<MT;n++) {\n'
        '      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;\n'
        '#pragma unroll\n'
        '      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));\n'
        '    }\n'
        '  }\n'
        '  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}\n'
        '  __syncthreads();\n'
        '  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}\n'
        '  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");\n'
        '  float c[2][MT][4]={};\n'
        '#pragma unroll\n'
        '  for(int k=0;k<KU;k++) {\n'
        '#pragma unroll\n'
        '    for(int n=0;n<MT;n++) {\n'
        '      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};\n'
        '#pragma unroll\n'
        '      for(int q=0;q<4;q++) {\n'
        '#pragma unroll\n'
        '        for(int h=0;h<2;h++) {\n'
        '          const int4 a=w[k][q+h*4];\n'
        '          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '#pragma unroll\n'
        '  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);\n'
        '  __syncthreads();\n'
        '  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");\n'
        '  if(tid<NI) {\n'
        '    float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);\n'
        '    const int dst=tid/OWN,local=tid%OWN;\n'
        '    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));\n'
        '  }\n'
        '\n'
        '  if(rank<P && tid<OWN) {\n'
        '    routertail_wait(rb,0);\n'
        '    float4 sum=make_float4(0,0,0,0);\n'
        '#pragma unroll\n'
        '    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);\n'
        '    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);\n'
        '    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}\n'
        '    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}\n'
        '  }\n'
        '\n'
        '}\n'
        '\n'
        'template<int MT,int S,int WK>\n'
        'int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  auto fn=routertail_small_kernel<MT,S,WK>;\n'
        '  constexpr int bytes=(S*1/4+WK)*MT*512+16;\n'
        '  static bool configured=false;\n'
        '  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}\n'
        '  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}\n'
        '  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;\n'
        '  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;\n'
        '  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);\n'
        '}\n'
        'int routertail_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {\n'
        '#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return routertail_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return routertail_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return routertail_small_t<4,S,W>(w,x,out,m,stream,query);}\n'
        'RSM(8,8) RSM(16,8) RSM(16,4)\n'
        'return -99;\n'
        '}\n'
        '\n'
    )

    rt__CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

    def rt__pack(w, s=16, wk=8):
        import torch
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)

        def units(v):
            v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
            return v.permute(0, 1, 3, 2, 4)
        data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
        ku = 128 // s // wk
        return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

    def rt_prepare_routertail_candidates(weights) -> dict:
        import torch
        from torch.utils.cpp_extension import load_inline
        w = weights.get('router')
        if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
            return {'enabled': False}
        if torch.cuda.get_device_capability(w.device) != (9, 0):
            return {'enabled': False}
        name = 'routertail_' + hashlib.sha256(rt__CUDA.encode()).hexdigest()[:12]
        ext = load_inline(name, rt__CPP, cuda_sources=rt__CUDA, functions=['routertail_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
        st = {'enabled': True, 'ext': ext, 'weight': w.contiguous(), 'packed': rt__pack(w)}
        for m in (8, 16, 17):
            x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
            rt_run_routertail_candidate(x, st)
        torch.cuda.synchronize(w.device)
        return st

    def rt_run_routertail_candidate(x, st):
        import torch
        if not st.get('enabled') or x.ndim != 2 or x.shape[1] != 8192 or (not 2 <= len(x) <= 32):
            return None
        if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != st['weight'].device:
            return None
        if not x.is_contiguous() or x.data_ptr() % 16:
            return None
        out = torch.empty((len(x), 256), device=x.device, dtype=torch.float32)
        rc = st['ext'].routertail_small(st['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 8, torch.cuda.current_stream().cuda_stream, 0)
        if rc:
            raise RuntimeError(f'routertail launch failed: {rc}')
        return out

    dd_DECODEDIET_ROUTES = (('pooler', 16), ('unpooler', 16))

    V18S_CUDA_TEMPLATE = (
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '#include <algorithm>\n'
        '#include <type_traits>\n'
        '#include <cooperative_groups.h>\n'
        '\n'
        '#define SM3D __device__ __forceinline__\n'
        '\n'
        'SM3D int4 v18s_ldw(const int4* p) {\n'
        '  int4 v;\n'
        '  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"\n'
        '               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SM3D uint4 v18s_ldx(const void* p) {\n'
        '  uint4 v;\n'
        '  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SM3D uint32_t v18s_lde(const uint32_t* p) {\n'
        '  uint32_t v;\n'
        '  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));\n'
        '  return v;\n'
        '}\n'
        '\n'
        'SM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {\n'
        '  uint32_t d;\n'
        '  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {\n'
        '  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n'
        '      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n'
        '      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));\n'
        '}\n'
        '\n'
        '// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.\n'
        'SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {\n'
        '  const uint32_t cm = C & 0x77777777u;\n'
        '  const uint32_t hlo = v18s_prmt(T0, T1, cm);\n'
        '  const uint32_t hhi = v18s_prmt(T0, T1, cm >> 16);\n'
        '  a[0] = v18s_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);\n'
        '  a[1] = v18s_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);\n'
        '  a[2] = v18s_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);\n'
        '  a[3] = v18s_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);\n'
        '}\n'
        '\n'
        '__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }\n'
        '\n'
        '// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.\n'
        'SM3D void v18s_order_point(int m, void* p) {\n'
        '  uint32_t t;\n'
        '  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));\n'
        '  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);\n'
        '}\n'
        '\n'
        'SM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
        '\n'
        'SM3D uint32_t v18s_rank() {\n'
        '  uint32_t r;\n'
        '  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n'
        '  return r;\n'
        '}\n'
        '\n'
        'SM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {\n'
        '  uint32_t d;\n'
        '  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));\n'
        '  return d;\n'
        '}\n'
        '\n'
        'SM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {\n'
        '  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");\n'
        '}\n'
        '\n'
        'SM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {\n'
        '  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");\n'
        '}\n'
        '\n'
        'SM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }\n'
        'SM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }\n'
        'SM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }\n'
        '\n'
        'SM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {\n'
        '  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"\n'
        '               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");\n'
        '}\n'
        '\n'
        'SM3D bool v18s_mbar_try(uint32_t bar) {\n'
        '  uint32_t ok;\n'
        '  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\\n"\n'
        '               " selp.u32 %0, 1, 0, p;\\n}" : "=r"(ok) : "r"(bar) : "memory");\n'
        '  return ok != 0;\n'
        '}\n'
        '\n'
        'SM3D uint64_t v18s_gtime() {\n'
        '  uint64_t t;\n'
        '  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n'
        '  return t;\n'
        '}\n'
        '\n'
        '// Bounded: a broken transaction count shows up as a wrong result instead of a hang.\n'
        'SM3D void v18s_wait_bar(uint32_t bar) {\n'
        '  if (v18s_mbar_try(bar)) return;\n'
        '  const uint64_t t0 = v18s_gtime();\n'
        '  while (!v18s_mbar_try(bar))\n'
        '    if (v18s_gtime() - t0 > 20000000ull) __trap();\n'
        '}\n'
        '\n'
        'SM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }\n'
        '\n'
        'SM3D float4 v18s_shfl_xor4(const float4& v, int o) {\n'
        '  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),\n'
        '                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));\n'
        '}\n'
        '\n'
        'SM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }\n'
        'SM3D void v18s_store(float* p, float v) { *p = v; }\n'
        '\n'
        'template <int FMT> struct Sm3Fmt;\n'
        'template <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };\n'
        'template <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };\n'
        'template <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };\n'
        'template <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };\n'
        '\n'
        'template <int FMT>\n'
        'SM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,\n'
        '                   uint32_t U0, uint32_t U1) {\n'
        '  if constexpr (FMT == 0 || FMT == 2) {\n'
        '#pragma unroll\n'
        '    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {\n'
        '      A[4 * l + 0] = (uint32_t)w[l].x;\n'
        '      A[4 * l + 1] = (uint32_t)w[l].y;\n'
        '      A[4 * l + 2] = (uint32_t)w[l].z;\n'
        '      A[4 * l + 3] = (uint32_t)w[l].w;\n'
        '    }\n'
        '  } else {\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {\n'
        '      const int4* u = w + 3 * h;\n'
        '      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;\n'
        '      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);\n'
        '      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);\n'
        '      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);\n'
        '      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        "// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.\n"
        'template <int NR>\n'
        'SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {\n'
        '  const uint32_t reg = (e >> 8) & 31u;\n'
        '  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);\n'
        '#pragma unroll\n'
        '  for (int r = 0; r < NR; ++r)\n'
        '    if (reg == (uint32_t)r) A[r] ^= xr;\n'
        '}\n'
        '\n'
        "// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.\n"
        'template <class Emit>\n'
        'SM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {\n'
        '  for (int idx = tid; idx < NI; idx += nt) {\n'
        '    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);\n'
        '    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);\n'
        '    emit(idx, v);\n'
        '  }\n'
        '}\n'
        '\n'
        '// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,\n'
        '// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB\n'
        '// straight-line code block at the end, 128 (with 4) no output stores.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n'
        '__global__ void __launch_bounds__(NTMAX) v18s_gemv(\n'
        '    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,\n'
        '    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,\n'
        '    uint32_t U0, uint32_t U1) {\n'
        '  const int m = CM ? CM : m_arg;\n'
        '  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;\n'
        '  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;\n'
        '  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;\n'
        '  constexpr bool C12 = FMT == 1 || FMT == 3;\n'
        '  constexpr bool HILO = FMT >= 2;\n'
        '  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;\n'
        '  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;\n'
        '  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;\n'
        '  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;\n'
        '  OutT* out = reinterpret_cast<OutT*>(outp);\n'
        '  extern __shared__ __align__(16) unsigned char smem[];\n'
        '  __shared__ __align__(8) uint64_t s_bar;\n'
        '  if constexpr (EMPTY) {\n'
        '    return;\n'
        '  } else {\n'
        '    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n'
        '    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;\n'
        '    const int S = CS ? CS : gridDim.y;\n'
        '    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;\n'
        '    const int G = GX * S * NW;\n'
        '    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;\n'
        '    const int rt0 = (blockIdx.x * WR + wr) * RT;\n'
        '    const int c0 = (blockIdx.y * WK + wk) * KU;\n'
        '    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);\n'
        '    const uint32_t bar = v18s_smem_u32(&s_bar);\n'
        '    uint32_t rank = 0, my_e = 0xFFFFFFFFu;\n'
        '    if constexpr (!(OPT & 1)) {\n'
        '    if constexpr (PUSH) {\n'
        '      rank = v18s_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        v18s_mbar_init(bar, 1);\n'
        '        v18s_mbar_expect(bar, parts * RT * TV * 128u);\n'
        '        v18s_fence_init();\n'
        '      }\n'
        '      v18s_cl_arrive();\n'
        '    }\n'
        '\n'
        '    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    }\n'
        '\n'
        '    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;\n'
        '    const size_t sstride = (size_t)G * (RT * NL * 32);\n'
        '    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;\n'
        '\n'
        '    int4 w[ST][RT][NL];\n'
        '    uint4 xv[ST][MT][2];\n'
        '    auto load = [&](int slot, int s) {\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);\n'
        '      if constexpr (!NOX && !RO && !(OPT & 2)) {\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n) {\n'
        '          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;\n'
        '          xv[slot][n][0] = v18s_ldx(xp);\n'
        '          xv[slot][n][1] = v18s_ldx(xp + 8);\n'
        '        }\n'
        '      }\n'
        '    };\n'
        '#pragma unroll\n'
        '    for (int s = 0; s < ST; ++s) load(s, s);\n'
        '    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n'
        '    if constexpr (OPT & 1) {\n'
        '    if constexpr (PUSH) {\n'
        '      rank = v18s_rank();\n'
        '      if (rank == 0 && tid == 0) {\n'
        '        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);\n'
        '        v18s_mbar_init(bar, 1);\n'
        '        v18s_mbar_expect(bar, parts * RT * TV * 128u);\n'
        '        v18s_fence_init();\n'
        '      }\n'
        '      v18s_cl_arrive();\n'
        '    }\n'
        '\n'
        '    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);\n'
        '\n'
        '    }\n'
        '\n'
        '    float acc[NA][RT][MT][4];\n'
        '#pragma unroll\n'
        '    for (int h = 0; h < NA; ++h)\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;\n'
        '    uint32_t fold = 0;\n'
        '    uint32_t nxt = 0xFFFFFFFFu;\n'
        '    int ep = 0;\n'
        '    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);\n'
        '\n'
        '#pragma unroll 1\n'
        '    for (int i = 0; i < KU; i += ST) {\n'
        '#pragma unroll\n'
        '      for (int s = 0; s < ST; ++s) {\n'
        '        const int si = i + s;\n'
        '        if (ST == 3 && si >= KU) break;\n'
        '        if constexpr (RO) {\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int l = 0; l < NL; ++l)\n'
        '              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n'
        '        } else {\n'
        '          uint32_t b[MT][8];\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n) {\n'
        '            if constexpr (NOX) {\n'
        '#pragma unroll\n'
        '              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;\n'
        '            } else if constexpr (OPT & 2) {\n'
        '              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;\n'
        '              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);\n'
        '              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;\n'
        '              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;\n'
        '            } else {\n'
        '              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;\n'
        '              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;\n'
        '            }\n'
        '          }\n'
        '          uint32_t A[RT][NA * 16];\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);\n'
        '          if (si + ST < KU) load(s, si + ST);\n'
        '          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);\n'
        '          if constexpr (C12) {\n'
        '#pragma unroll\n'
        '            for (int j = 0; j < RT; ++j) {\n'
        '              const uint32_t useq = (uint32_t)(si * RT + j);\n'
        '              if ((nxt >> 18) == useq) {\n'
        '                do {\n'
        '                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);\n'
        '                  ++ep;\n'
        '                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;\n'
        '                } while ((nxt >> 18) == useq);\n'
        '              }\n'
        '            }\n'
        '          }\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '              for (int q = 0; q < 4; ++q) {\n'
        '                if constexpr (NOMMA) {\n'
        '#pragma unroll\n'
        '                  for (int h = 0; h < NA; ++h)\n'
        '                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^\n'
        '                            A[j][16 * h + 4 * q + 3];\n'
        '                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];\n'
        '                } else {\n'
        '                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);\n'
        '                }\n'
        '              }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '\n'
        '    if constexpr (RO) {\n'
        '      if constexpr (RO_CODE) {\n'
        '        if (warp == 0) {\n'
        '          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;\n'
        '          const uint32_t k = (uint32_t)m | 1u;\n'
        '#pragma unroll\n'
        '          for (int q = 0; q < 64; ++q)\n'
        '            asm volatile("mad.lo.u32 %0, %0, %4, %5;\\n mad.lo.u32 %1, %1, %4, %5;\\n mad.lo.u32 %2, %2, %4, %5;\\n"\n'
        '                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));\n'
        '          fold = a ^ b ^ c ^ d;\n'
        '        }\n'
        '      }\n'
        '      if constexpr (RO_STORE) {\n'
        '        if ((lane & 3) == 0)\n'
        '          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));\n'
        '      } else {\n'
        '        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);\n'
        '      }\n'
        '      return;\n'
        '    } else {\n'
        '      float r[RT][MT][4];\n'
        '#pragma unroll\n'
        '      for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '        for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];\n'
        '      if constexpr (NOMMA) {\n'
        '        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;\n'
        '      }\n'
        '\n'
        '      const int RB = WR * RT * 16;\n'
        '      const int row0 = blockIdx.x * RB;\n'
        '      if (NORED || (RED == 0 && WK == 1)) {\n'
        '#pragma unroll\n'
        '        for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '          for (int n = 0; n < MT; ++n)\n'
        '#pragma unroll\n'
        '            for (int c = 0; c < 4; ++c) {\n'
        '              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);\n'
        '              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);\n'
        '              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)\n'
        '                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);\n'
        '            }\n'
        '        return;\n'
        '      }\n'
        '      if constexpr (!NORED) {\n'
        '        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.\n'
        '        auto emit_full = [&](int idx, float4 v) {\n'
        '          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);\n'
        '            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);\n'
        '            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        auto emit_compact = [&](int idx, float4 v) {\n'
        '          if constexpr (OPT & 8) { emit_full(idx, v); return; }\n'
        '          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;\n'
        '          const float vv[4] = {v.x, v.y, v.z, v.w};\n'
        '#pragma unroll\n'
        '          for (int c = 0; c < 4; ++c) {\n'
        '            const int tok = 2 * t + (c & 1);\n'
        '            const int row = row0 + tile * 16 + g + 8 * (c >> 1);\n'
        '            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);\n'
        '          }\n'
        '        };\n'
        '        if constexpr (RED == 0 || RED == 1) {\n'
        '          float4* part = reinterpret_cast<float4*>(smem);\n'
        '          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;\n'
        '#pragma unroll\n'
        '          for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '            for (int n = 0; n < MT; ++n)\n'
        '              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)\n'
        '              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =\n'
        '                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '          __syncthreads();\n'
        '          if constexpr (RED == 0) {\n'
        '            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });\n'
        '          } else {\n'
        '            float4* csum = part + (size_t)NW * RT * MT * 32;\n'
        '            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });\n'
        '            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();\n'
        '            cl.sync();\n'
        '            const int rk = (int)cl.block_rank();\n'
        '            const int per = (NI + S - 1) / S;\n'
        '            const int e1 = min(NI, (rk + 1) * per);\n'
        '            for (int idx = rk * per + tid; idx < e1; idx += nt) {\n'
        '              float4 v = *cl.map_shared_rank(csum + idx, 0);\n'
        '              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));\n'
        '              emit_full(idx, v);\n'
        '            }\n'
        '            cl.sync();\n'
        '          }\n'
        '        } else {\n'
        '          const int NIc = WR * RT * TV * 8;\n'
        '          const int t = lane & 3, g = lane >> 2;\n'
        '          float4* recv = reinterpret_cast<float4*>(smem);\n'
        "          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.\n"
        '          if constexpr (RED == 2) {\n'
        '            v18s_cl_wait();\n'
        '            if (t < TV) {\n'
        '              const uint32_t rbar = v18s_mapa(bar, 0);\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n) {\n'
        '                  const int tn = n * 4 + t;\n'
        '                  if (tn < TV) {\n'
        '                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);\n'
        '                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;\n'
        '                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '                    if (rank == 0) *dst = v;\n'
        '                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);\n'
        '                  }\n'
        '                }\n'
        '            }\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            v18s_wait_bar(bar);\n'
        '            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);\n'
        '          } else {\n'
        '            float4* part = recv + (size_t)S * NIc;\n'
        '            if (t < TV) {\n'
        '#pragma unroll\n'
        '              for (int j = 0; j < RT; ++j)\n'
        '#pragma unroll\n'
        '                for (int n = 0; n < MT; ++n)\n'
        '                  if (n * 4 + t < TV)\n'
        '                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =\n'
        '                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);\n'
        '            }\n'
        '            __syncthreads();\n'
        '            v18s_cl_wait();\n'
        '            const uint32_t rbar = v18s_mapa(bar, 0);\n'
        '            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {\n'
        '              float4* dst = recv + (size_t)rank * NIc + idx;\n'
        '              if (rank == 0) *dst = v;\n'
        '              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);\n'
        '            });\n'
        '            if (rank != 0) return;\n'
        '            __syncthreads();\n'
        '            v18s_wait_bar(bar);\n'
        '            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);\n'
        '          }\n'
        '        }\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'static int v18s_smem_optin() {\n'
        '  static int v = -1;\n'
        '  if (v < 0) {\n'
        '    int dev = 0;\n'
        '    cudaGetDevice(&dev);\n'
        '    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);\n'
        '  }\n'
        '  return v;\n'
        '}\n'
        '\n'
        '// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.\n'
        'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n'
        'static int v18s_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  auto kern = v18s_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;\n'
        '  if (query == 2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;\n'
        '    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);\n'
        '  }\n'
        '  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;\n'
        '  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;\n'
        '  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;\n'
        '  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;\n'
        '  if (N % RB) return -2;\n'
        '  if ((int64_t)S * WK * KU * 64 != K) return -3;\n'
        '  if ((ST != 3 && KU % ST) || KU < ST) return -4;\n'
        '  if (m < 1 || m > MT * 8) return -5;\n'
        '  const bool cl = RED != 0;\n'
        '  if (!cl && S != 1) return -9;\n'
        '  if (cl && (S < 2 || S > 16)) return -9;\n'
        '  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);\n'
        '  size_t sm = 0;\n'
        '  if (!(DIAG & 28)) {\n'
        '    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;\n'
        '    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;\n'
        '    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;\n'
        '    else sm = (size_t)(S + WK) * WR * RT * TV * 128;\n'
        '  }\n'
        '  static int max_dyn = -1;\n'
        '  if (max_dyn < 0) {\n'
        '    cudaFuncAttributes fa;\n'
        '    cudaFuncGetAttributes(&fa, kern);\n'
        '    max_dyn = v18s_smem_optin() - (int)fa.sharedSizeBytes;\n'
        '    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);\n'
        '    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n'
        '  }\n'
        '  if ((int64_t)sm > max_dyn) return -6;\n'
        '  cudaLaunchConfig_t lc = {};\n'
        '  cudaLaunchAttribute at[1];\n'
        '  lc.gridDim = dim3(N / RB, S, 1);\n'
        '  lc.blockDim = dim3(nt, 1, 1);\n'
        '  lc.dynamicSmemBytes = sm;\n'
        '  lc.stream = (cudaStream_t)stream;\n'
        '  if (cl) {\n'
        '    at[0].id = cudaLaunchAttributeClusterDimension;\n'
        '    at[0].val.clusterDim.x = 1;\n'
        '    at[0].val.clusterDim.y = S;\n'
        '    at[0].val.clusterDim.z = 1;\n'
        '    lc.attrs = at;\n'
        '    lc.numAttrs = 1;\n'
        '  }\n'
        '  if (query == 1) {\n'
        '    int v = 0;\n'
        '    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)\n'
        '                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);\n'
        '    cudaGetLastError();\n'
        '    return e == cudaSuccess ? v : -11;\n'
        '  }\n'
        '  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),\n'
        '                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,\n'
        '                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),\n'
        '                     (uint32_t)((uint64_t)u01 >> 32));\n'
        '  return cudaGetLastError() == cudaSuccess ? 0 : -8;\n'
        '}\n'
        '\n'
        'int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n'
        '              int S, int64_t t01, int64_t u01, int64_t stream, int query) {\n'
        '  switch (inst) {\n'
        '@@CASES@@\n'
        '    default: return -99;\n'
        '  }\n'
        '}\n'
        '\n'
        '#include <cuda_runtime.h>\n'
        '#include <cuda_bf16.h>\n'
        '#include <stdint.h>\n'
        '\n'
        'template<int M, int NT, int ST>\n'
        '__global__ void v18s_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,\n'
        '                                float* __restrict__ y, int m) {\n'
        '  constexpr int K=8192;\n'
        '  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;\n'
        '  float acc[M][4]={};\n'
        '  float4 ww[ST];\n'
        '  uint2 xx[ST][M];\n'
        '  auto ld=[&](int slot,int i) {\n'
        '    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) {\n'
        '      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];\n'
        '    }\n'
        '  };\n'
        '  #pragma unroll\n'
        '  for(int s=0;s<ST;s++) ld(s,tid+s*NT);\n'
        '  #pragma unroll 1\n'
        '  for(int base=tid;base<K/4;base+=ST*NT) {\n'
        '    #pragma unroll\n'
        '    for(int s=0;s<ST;s++) {\n'
        '      float4 v=ww[s];\n'
        '      uint2 a[M];\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) a[t]=xx[s][t];\n'
        '      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);\n'
        '      #pragma unroll\n'
        '      for(int t=0;t<M;t++) if(t<m) {\n'
        '        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);\n'
        '        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);\n'
        '        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);\n'
        '        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);\n'
        '        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);\n'
        '        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);\n'
        '        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '  __shared__ float sums[M][NT/32];\n'
        '  #pragma unroll\n'
        '  for(int t=0;t<M;t++) if(t<m) {\n'
        '    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);\n'
        '    #pragma unroll\n'
        '    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '    if(lane==0) sums[t][warp]=v;\n'
        '  }\n'
        '  __syncthreads();\n'
        '  if(warp==0) {\n'
        '    #pragma unroll\n'
        '    for(int t=0;t<M;t++) if(t<m) {\n'
        '      float v=lane<NT/32?sums[t][lane]:0.f;\n'
        '      #pragma unroll\n'
        '      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);\n'
        '      if(lane==0) y[t*256+row]=v;\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'int v18s_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {\n'
        '  auto kern=v18s_router_scalar<1,512,2>;\n'
        '  if(query==2) {\n'
        '    cudaFuncAttributes fa;\n'
        '    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;\n'
        '    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);\n'
        '  }\n'
        '  if(m!=1) return -5;\n'
        '  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);\n'
        '  return cudaGetLastError()==cudaSuccess?0:-8;\n'
        '}\n'
    )

    V18S_CPP = '\nint v18s_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

    V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

    V18S_ESC_CAP = 32

    def v18s_cuda_source(instances):
        cases = []
        for i, inst in enumerate(instances):
            args = ", ".join(str(v) for v in (inst + (0,) * (15 - len(inst))))
            cases.append(f"    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                         " stream, query);")
        return V18S_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))

    def v18s_build(instances, name="v18s_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
        return load_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda,
                           functions=["v18s_launch", "v18s_router_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                           no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

    def v18s_units(w):
        """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
        return v.reshape(n // 16, k // 64, 32, 32)

    def v18s_to_i16(v):
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

    def v18s_to_i32(v):
        return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

    def v18s_bf16_bytes(v):
        r, c = v.shape[:2]
        words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
        return words.view(torch.uint8).view(r, c, 2048)

    def v18s_c12_bytes(v):
        """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
        r, c = v.shape[:2]
        dev = v.device
        lo = v & 0xFF
        sign = v >> 15
        eh = (v >> 8) & 0x7F
        hist = torch.bincount(eh.flatten(), minlength=128).cpu()
        window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
        base = int(window.argmax())
        d = eh - base
        inwin = (d >= 0) & (d < 8)
        d = torch.where(inwin, d, torch.zeros_like(d))
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
        code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
        code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
        lo8 = lo.to(torch.uint8)
        data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
        er, ec, el, ev = (~inwin).nonzero().unbind(1)
        xor7 = eh[er, ec, el, ev] ^ base
        return data, base, (er, ec, el, ev, xor7)

    def v18s_c12_unpack(data, base, esc):
        """Inverse of v18s_c12_bytes, used to prove the packing is exact."""
        r, c = data.shape[:2]
        dev = data.device
        p = data.view(r, c, 3, 32, 16)
        lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
        code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
        nib = (torch.arange(8, device=dev) * 4).long()
        pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
        d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
        sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
        val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
        er, ec, el, ev, xor7 = esc
        val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
        return val

    def v18s_table(base):
        t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
        t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
        v = t0 | t1 << 32
        return v - 2**64 if v >= 2**63 else v

    def v18s_pack_format(w, fmt):
        """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

        Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
        half 1 marks the lo matrix of a HILO pair."""
        if fmt in (2, 3):
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            mats = [v18s_units(hi), v18s_units(lo)]
        else:
            mats = [v18s_units(w)]
        if fmt in (0, 2):
            data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
            return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
        parts, tabs, escs = [], [], []
        for h, v in enumerate(mats):
            data, base, esc = v18s_c12_bytes(v)
            if not torch.equal(v18s_c12_unpack(data, base, esc), v):
                return None
            parts.append(data)
            tabs.append(v18s_table(base))
            escs.append(esc + (torch.full_like(esc[0], h),))
        esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
        return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
                "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

    def v18s_geometry(R, C, rt, wr, wk, ku, s):
        if R % (wr * rt) or C != s * wk * ku:
            return None
        gx = R // (wr * rt)
        return gx, wr * wk, gx * s * wr * wk

    def v18s_stream(pk, rt, wr, wk, ku, s):
        """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
        data = pk["data"]
        R, C, B = data.shape
        geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
        if geo is None:
            return None
        gx, nw, G = geo
        dev = data.device
        si = torch.arange(ku, device=dev).view(ku, 1, 1)
        gw = torch.arange(G, device=dev).view(1, G, 1)
        j = torch.arange(rt, device=dev).view(1, 1, rt)
        cta, warp = gw // nw, gw % nw
        by, bx = cta // gx, cta % gx
        wri, wki = warp % wr, warp // wr
        tile = (bx * wr + wri) * rt + j
        chunk = (by * wk + wki) * ku + si
        idx = (tile * C + chunk).reshape(-1)
        wp = data.view(R * C, B)[idx].contiguous().view(-1)
        esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
        if pk["esc"] is not None and pk["esc"][0].numel():
            er, ec, el, ev, x7, half = pk["esc"]
            bx_e, rem = er // (wr * rt), er % (wr * rt)
            wr_e, j_e = rem // rt, rem % rt
            by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
            wk_e, s_e = rem2 // ku, rem2 % ku
            g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
            seq = s_e * rt + j_e
            if int(seq.max()) >= (1 << 14) - 1:
                return None
            counts = torch.bincount(g_e, minlength=G)
            if int(counts.max()) > V18S_ESC_CAP:
                return None
            order = torch.argsort(g_e * (1 << 14) + seq)
            g_s, seq_s = g_e[order], seq[order]
            rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
            reg = ev[order] // 2 + 16 * half[order]
            entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
                (ev[order] % 2).long() << 7) | x7[order].long()
            esc[g_s, rank] = v18s_to_i32(entry)
        return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
                "max_esc": int((esc != -1).sum(1).max())}

    def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
        """cfg = (WR, WK, KU, S)."""
        if sp["fmt"] == 4:
            return ext.v18s_router_launch(sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                         torch.cuda.current_stream().cuda_stream, query)
        wr, wk, ku, s = cfg
        m, k = x.shape
        return ext.v18s_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                             wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

    def v18s_rows(layer, m, n_full):
        return 1024 if layer == "unpooler" and m <= 8 else n_full

    def v18s_bf16_twin(inst):
        """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
        return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

    def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
        """Runs every column as a one-hot input: each output must equal its weight column exactly.

        exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
        n, k = exact.shape
        mm = rows_per_launch
        dev = exact.device
        out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
        got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
        eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
        for i in range(k // mm):
            if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
                return False
        return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

    def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
        """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
        n, k = w_rows.shape
        is_router = w_rows.dtype == torch.float32
        tol = 1e-4 if is_router else 1e-2
        gen = torch.Generator(device=w_rows.device).manual_seed(seed)
        for m in ms:
            x = torch.randn((m, k), generator=gen, device=w_rows.device)
            x[m - 1].zero_()
            x[m - 1, (seed * 131 + m) % k] = 16.0
            x = x.to(torch.bfloat16)
            ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
            out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
            if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
                return False
            diff = out.double() - ref
            rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
            if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                    float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
                return False
        return True

    def v18s_exact(w_rows, fmt):
        if fmt in (2, 3):
            hi = w_rows.to(torch.bfloat16)
            return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
        return w_rows

    def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
        """Packs, stream-orders and verifies one table entry; None if any step fails."""
        m_hi = key[2] if key[1] == "m" else 8 * inst[1]
        rows = v18s_rows(layer, m_hi, w.shape[0])
        fmt = inst[0]
        pkey = (layer, rows, fmt)
        if pkey not in packs:
            packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
        if packs[pkey] is None:
            return None
        skey = pkey + (inst[2],) + tuple(cfg)
        if skey not in streams:
            streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
        sp = streams[skey]
        if sp is None:
            return None
        vkey = ("verified", inst) + skey
        if vkey not in streams:
            w_rows = w[:rows].contiguous()
            lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
            fixed_m = inst[7] if len(inst) > 7 else 0
            ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
            streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
                v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
        if not streams[vkey]:
            return None
        return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

    V18_SMALL = {('pooler', 1): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1)), ('o', 8): ((1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))}


    _SLIM_OLAYER_CONFIGS = [(128, 128, 4, 3, 1, 1)]
    _SLIM_V18_O8 = V18_SMALL[('o', 8)]


    def prepare(weights):
        state = {'weights': weights}
        state['router32'] = prepare_router({'router': weights['router']})
        state['router_tail'] = rt_prepare_routertail_candidates(
            {'router': weights['router']})

        state['olayer'] = prepare_olayer_candidates(
            {'o': weights['o']}, _SLIM_OLAYER_CONFIGS)
        x = torch.zeros((128, 1024), dtype=torch.bfloat16,
                        device=weights['o'].device)
        run_olayer_candidate(x, state['olayer'], 0)

        state['o256'] = prepare_v10_tune({'o': weights['o']})
        x = torch.zeros((256, 1024), dtype=torch.bfloat16,
                        device=weights['o'].device)
        run_v10_tune(x, 'o', 8192, state['o256'])

        state['pooler8'] = tt_v10_sm2_prepare_smallm2(
            {'pooler': weights['pooler']})
        state['unpooler8'] = tt_v13_smtrace8_prepare_smtrace(
            {'unpooler': weights['unpooler']})
        state['qkv128'] = tt_v14_qkv_prepare_qkvbig(
            {'qkv_gate': weights['qkv_gate']})
        state['dense16'] = dd_prepare_decodediet({
            'pooler': weights['pooler'], 'unpooler': weights['unpooler']})

        inst, cfg = _SLIM_V18_O8
        instances = list(dict.fromkeys((inst, v18s_bf16_twin(inst))))
        ext = v18s_build(instances, 'v18_small_selected')
        ids = {value: index for index, value in enumerate(instances)}
        plan = v18s_plan(ext, ids, {}, {}, weights['o'], 'o', ('o', 'm', 8),
                         inst, cfg)
        if plan is None:
            twin = v18s_bf16_twin(inst)
            plan = v18s_plan(ext, ids, {}, {}, weights['o'], 'o',
                             ('o', 'm', 8), twin, cfg)
        state['o8'] = {'ext': ext, 'plan': plan}
        torch.cuda.synchronize(weights['qkv_gate'].device)
        return state


    def _fallback(data, state):
        x, cfg = data['x'], data['config']
        weight = state['weights'][cfg['layer']]
        if cfg['layer'] != 'router':
            return x @ weight[:cfg['n']].T
        if x.shape[0] <= 8:
            return F.linear(x.float(), weight)
        hi = weight.to(torch.bfloat16)
        out = torch.mm(x, hi.T, out_dtype=torch.float32)
        out += torch.mm(x, (weight - hi.float()).to(torch.bfloat16).T,
                        out_dtype=torch.float32)
        return out


    def kernel_fn(data, state):
        x, cfg = data['x'], data['config']
        layer, m, n = cfg['layer'], cfg['m'], cfg['n']
        out = None
        if layer == 'o' and m == 8:
            plan = state['o8']['plan']
            if plan is not None:
                out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
                rc = v18s_launch_raw(state['o8']['ext'], plan['inst_id'], plan,
                                     x, out, n, plan['cfg'])
                if rc:
                    out = None
        elif layer == 'o' and m == 128:
            out = run_olayer_candidate(x, state['olayer'], 0)
        elif layer == 'o' and m == 256:
            out = run_v10_tune(x, layer, n, state['o256'])
        elif layer == 'pooler' and m == 8:
            out = tt_v10_sm2_run_smallm2(x, layer, n, state['pooler8'])
        elif layer == 'unpooler' and m == 8:
            out = tt_v13_smtrace8_run_smtrace(x, layer, n, state['unpooler8'])
        elif layer in ('pooler', 'unpooler') and m == 16:
            out = dd_run_decodediet(x, layer, n, state['dense16'])
        elif layer == 'qkv_gate' and m == 128:
            out = tt_v14_qkv_run_qkvbig(x, layer, n, state['qkv128'])
        elif layer == 'router' and m in (8, 16):
            out = rt_run_routertail_candidate(x, state['router_tail'])
        elif layer == 'router' and m == 32:
            out = run_router(x, layer, n, state['router32'])
        return _fallback(data, state) if out is None else out

    return _types.SimpleNamespace(**locals())

_alt = _v179_build_alt_namespace()
del _v179_build_alt_namespace
_main_prepare, _main_kernel_fn = prepare, kernel_fn


def prepare(weights):
    state = _main_prepare(weights)
    if _ALT_ROUTES:
        state["_alt"] = _alt.prepare(weights)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["layer"], cfg["m"]) in _ALT_ROUTES and "_alt" in state:
        return _alt.kernel_fn(data, state["_alt"])
    return _main_kernel_fn(data, state)
# v102: QKV128 half-width BF16 DSMEM epilogue.
#
# RED=9 leaves the protected WGMMA mainloop, split-K geometry, token ownership,
# and owner-side rank order unchanged.  Each sender rounds its remote FP32
# fragment to four BF16 values and sends one 8-byte DSMEM transaction instead
# of one 16-byte FP32 transaction.  The owner expands remote BF16 fragments,
# adds ranks 0 -> 1 -> 2 in FP32, rounds once more for the final output, and
# uses the incumbent stmatrix/vector-store path. Following explicit user
# authorization to score the measured 2.742% win, dispatch remains on v87
# unless H200 correctness, exhaustive coefficient recovery, resource,
# residency, and a paired cold-L2 non-regression gate all pass.

import hashlib as _v102_hashlib
import statistics as _v102_statistics


_V102_Q128_CONFIGS = (
    # NR, output groups, token tile, split-K, stages, pipeline flags, reduction.
    (64, 36, 128, 3, 8, 539, 9),
)


def _v102_cuda_source(configs):
    """Generate the proven QKV128 core with only the RED=9 epilogue added."""
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = _alt.tt_v14_qkv_QKVBIG_CUDA

    helper_marker = 'QD uint64_t q_desc(uint32_t p)'
    if source.count(helper_marker) != 1:
        raise RuntimeError('v102 qkv source: helper marker mismatch')
    helper = r'''QD void q_push_bf16(uint32_t p,uint2 v,uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0],{%1,%2},[%3];"::
    "r"(p),"r"(v.x),"r"(v.y),"r"(b):"memory");
}
'''
    source = source.replace(helper_marker, helper + helper_marker, 1)

    old_kernel_sizes = r'''  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;'''
    new_kernel_sizes = r'''  constexpr int NV=NR*TM/4, OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int RSZ=RED==9?8:16;
  constexpr int RB=ST*SB+ST*16, RECV=RED==3?0:RB+16;'''
    if source.count(old_kernel_sizes) != 1:
        raise RuntimeError('v102 qkv source: kernel size marker mismatch')
    source = source.replace(old_kernel_sizes, new_kernel_sizes, 1)

    if source.count('q_expect(rb,(S-1)*own*16);') != 2:
        raise RuntimeError('v102 qkv source: barrier expectation mismatch')
    source = source.replace('q_expect(rb,(S-1)*own*16);',
                            'q_expect(rb,(S-1)*own*RSZ);')

    old_epilogue = '  } else if constexpr(RED>=4) {'
    new_epilogue = r'''  } else if constexpr(RED==9) {
    static_assert(NR==64 && TM==128 && S==3 && (PIPE&8));
    asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");
#pragma unroll
    for(int j=0;j<TM/8;j++) {
      int dst=j%S,local=(j/S)*128+tid,slot=rank-(rank>dst);
      if(dst!=rank) {
        __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
        __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
        q_push_bf16(q_map(base+RECV+(slot*OWN+local)*RSZ,dst),
          make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi),q_map(rb,dst));
      }
    }
    q_wait(rb,0);
#pragma unroll
    for(int j=0;j<TM/8;j++) if(j%S==rank) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) {
        float4 z;
        if(r==rank) z=make_float4(a[4*j],a[4*j+1],a[4*j+2],a[4*j+3]);
        else {
          uint2 p=*(uint2*)(sm+RECV+((r-(r>rank))*OWN+(j/S)*128+tid)*RSZ);
          __nv_bfloat162 lo=*(const __nv_bfloat162*)&p.x;
          __nv_bfloat162 hi=*(const __nv_bfloat162*)&p.y;
          float2 f0=__bfloat1622float2(lo),f1=__bfloat1622float2(hi);
          z=make_float4(f0.x,f0.y,f1.x,f1.y);
        }
        sum.x+=z.x;sum.y+=z.y;sum.z+=z.z;sum.w+=z.w;
      }
      a[4*j]=sum.x;a[4*j+1]=sum.y;a[4*j+2]=sum.z;a[4*j+3]=sum.w;
    }
    asm volatile("bar.sync 1,128;":::"memory");
#pragma unroll
    for(int j=0;j<TM/8;j++) if(j%S==rank) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rr=(tid/32)*16+((tid%16)/8)*8,cc=8*j+tid%8;
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"::
        "r"(base+2*(cc*72+rr)),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi):"memory");
    }
    asm volatile("bar.sync 1,128;":::"memory");
    for(int q=tid;q<(own/128)*64;q+=128) {
      int token=(rank+(q/64)*S)*8+(q/8)%8,rr=(q%8)*8;
      if(tok+token<m) *(uint4*)(out+(tok+token)*2304+row+rr)=
        *(uint4*)((__nv_bfloat16*)sm+token*72+rr);
    }
  } else if constexpr(RED>=4) {'''
    if source.count(old_epilogue) != 1:
        raise RuntimeError('v102 qkv source: epilogue marker mismatch')
    source = source.replace(old_epilogue, new_epilogue, 1)

    old_launch_sizes = r'''  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int KT=(PIPE&32)?2:1;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*16:0);'''
    new_launch_sizes = r'''  constexpr int LR=(2304+G-1)/G,NV=NR*TM/4,OWN=RED>=4?((NV/128+S-1)/S)*128:((PIPE&16)?(NV+S-1)/S:((NV+128*S-1)/(128*S))*128);
  constexpr int KT=(PIPE&32)?2:1,RSZ=RED==9?8:16;
  constexpr int bytes=ST*KT*(NR+TM)*128+ST*16+16+(S>1 && RED!=3?(RED>=4?S-1:S)*OWN*RSZ:0);'''
    if source.count(old_launch_sizes) != 1:
        raise RuntimeError('v102 qkv source: launch size marker mismatch')
    source = source.replace(old_launch_sizes, new_launch_sizes, 1)

    source = source.replace(
        '@@MMA@@', '\n'.join(_alt.tt_v14_qkv_qbig_mma(n) for n in ns))
    source = source.replace(
        '@@CALL@@',
        ' '.join(f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);'
                 for n in ns))
    source = source.replace(
        '@@SWAP@@',
        ' '.join(f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns))
    source = source.replace(
        '@@CASES@@',
        '\n'.join(
            f"case {i}: return qb_launch_t<{','.join(map(str, cfg))},false>"
            "(w,x,out,m,stream,query,stats);"
            for i, cfg in enumerate(configs)))
    return source


def _v102_build_qkv(configs):
    source = _v102_cuda_source(configs)
    digest = _v102_hashlib.sha256(source.encode()).hexdigest()[:16]
    namespace = 'q102_' + digest
    begin = source.index('#define QD')
    args = ('int id,int64_t w,int64_t x,int64_t out,int m,'
            'int64_t stream,int query,int64_t stats')
    source = (source[:begin] + f'namespace {namespace} {{\n' + source[begin:] +
              f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch('
              'id,w,x,out,m,stream,query,stats);}')
    cpp = ('#include <pybind11/pybind11.h>\n#include <stdint.h>\n'
           'int qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);')
    return load_inline(
        'qkv128_v102_' + digest, cpp, cuda_sources=source,
        functions=['qb_launch'], no_implicit_headers=True,
        with_pytorch_error_handling=False, verbose=False,
        extra_cuda_cflags=[
            '-O3', '-std=c++20', '-lineinfo',
            '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])


def _v102_prepare_q128(weights):
    w = weights['qkv_gate']
    configs = list(_V102_Q128_CONFIGS)
    return {
        'ext': _v102_build_qkv(configs),
        'configs': configs,
        'packed': {
            (groups, kt): _alt.tt_v14_qkv_qbig_pack(w, groups, kt)
            for groups, kt in {
                (cfg[1], 2 if cfg[5] & 32 else 1) for cfg in configs}},
        'device': w.device,
        'enabled': False,
    }


def _v102_run_q128(x, st):
    cfg = st['configs'][0]
    out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16,
                      device=x.device)
    packed = st['packed'][cfg[1], 2 if cfg[5] & 32 else 1]
    rc = st['ext'].qb_launch(
        0, packed.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
        torch.cuda.current_stream().cuda_stream, 0, 0)
    if rc:
        raise RuntimeError(f'v102 qkv128 launch {cfg} failed: {rc}')
    return out


def _v102_query_q128(x, st, query):
    cfg = st['configs'][0]
    out = torch.empty((128, 2304), dtype=torch.bfloat16, device=x.device)
    packed = st['packed'][cfg[1], 2 if cfg[5] & 32 else 1]
    return st['ext'].qb_launch(
        0, packed.data_ptr(), x.data_ptr(), out.data_ptr(), 128,
        torch.cuda.current_stream().cuda_stream, query, 0)


def _v102_random_check(st, w, x, ref, incumbent):
    out_bf16 = _v102_run_q128(x, st)
    bitwise = bool(torch.equal(out_bf16, incumbent))
    probe_exact = bool(torch.equal(out_bf16[-1], incumbent[-1]))
    out = out_bf16.double()
    torch.cuda.synchronize(w.device)
    diff = out - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    ok = (bool(torch.isfinite(out).all()) and probe_exact and
          rel <= 0.0075 and worst <= 0.03)
    return ok, bitwise, probe_exact, rel, worst


def _v102_exact_check(st, w):
    """Every one-hot column must recover the original BF16 coefficient."""
    m, k, n = 128, 8192, 2304
    rows = torch.arange(m, device=w.device)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
    got = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for base in range(0, k, m):
        x.zero_()
        x[rows, base + rows] = 1
        got[base:base + m].copy_(_v102_run_q128(x, st))
    torch.cuda.synchronize(w.device)
    return bool(torch.equal(got, w.t()))


def _v102_capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        held = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, held


def _v102_paired_graph_ms(candidate, incumbent, device, repeats=9):
    cg, candidate_out = _v102_capture(candidate)
    ig, incumbent_out = _v102_capture(incumbent)
    keep = (candidate_out, incumbent_out)
    flush = torch.empty((64 * 1024, 1024), dtype=torch.float32,
                        device=device)
    sink = torch.empty((1024,), dtype=torch.float32, device=device)
    events = []
    for rep in range(repeats):
        order = (('candidate', cg), ('incumbent', ig))
        if rep & 1:
            order = order[::-1]
        for name, graph in order:
            torch.sum(flush, dim=0, out=sink)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            events.append((name, start, end))
    torch.cuda.synchronize(device)
    times = {'candidate': [], 'incumbent': []}
    for name, start, end in events:
        times[name].append(start.elapsed_time(end))
    assert keep[0] is not None and keep[1] is not None
    return (_v102_statistics.median(times['candidate']),
            _v102_statistics.median(times['incumbent']))


_v102_base_prepare = prepare
_v102_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v102_base_prepare(weights)
    w = weights.get('qkv_gate')
    if (w is None or w.shape != (2304, 8192) or
            w.dtype != torch.bfloat16 or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        exp = _v102_prepare_q128(weights)
        gen = torch.Generator(device=w.device).manual_seed(102128)
        x = torch.randn((128, 8192), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        x[-1].zero_()
        x[-1, 7919] = 16
        ref = x.double() @ w.double().t()
        data = {'x': x, 'config': {
            'm': 128, 'n': 2304, 'k': 8192, 'layer': 'qkv_gate'}}
        incumbent = _v102_base_kernel_fn(data, state)
        occupancy = _v102_query_q128(x, exp, 1)
        attributes = _v102_query_q128(x, exp, 2)
        ok, bitwise, probe_exact, rel, worst = _v102_random_check(
            exp, w, x, ref, incumbent)
        print('v102 qkv128 bf16-dsmem', exp['configs'][0],
              'active_clusters', occupancy, 'attrs', attributes,
              'errors', rel, worst, 'bitwise_v87', bitwise,
              'probe_exact', probe_exact, 'valid', ok, flush=True)
        if not (ok and occupancy >= 36 and attributes >= 0 and
                not attributes % 1000):
            print('v102 qkv128 rejected: accuracy, spills, or one-wave residency',
                  flush=True)
            state['_v102_q128'] = exp
            return state

        cand_ms, base_ms = _v102_paired_graph_ms(
            lambda: _v102_run_q128(x, exp),
            lambda: _v102_base_kernel_fn(data, state), w.device)
        ratio = cand_ms / base_ms
        print('v102 qkv128 gate candidate', cand_ms * 1e3,
              'incumbent', base_ms * 1e3, 'ratio', ratio, flush=True)
        if ratio < 1.0 and _v102_exact_check(exp, w):
            exp['enabled'] = True
            print('v102 qkv128 accepted user-authorized half-width BF16 DSMEM',
                  flush=True)
        else:
            why = ('paired non-regression' if ratio >= 1.0
                   else 'exhaustive coefficient recovery')
            print('v102 qkv128 rejected:', why, flush=True)
        state['_v102_q128'] = exp
    except Exception as exc:
        print('v102 qkv128 experiment unavailable:', type(exc).__name__,
              str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    exp = state.get('_v102_q128')
    if (exp is not None and exp.get('enabled') and
            cfg['layer'] == 'qkv_gate' and cfg['m'] == 128 and
            cfg['n'] == 2304 and cfg['k'] == 8192):
        x = data['x']
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == exp['device'] and x.data_ptr() % 16 == 0):
            return _v102_run_q128(x, exp)
    return _v102_base_kernel_fn(data, state)


# v105: exact C12 GEMV K-partition/thread-count tournament for the scored m=1
# dense shapes.  ROWS remains 2, so the incumbent packed planes and escape
# tables are reused byte-for-byte.  Only NTH changes; protected v102 is the
# immediate fallback for every failed correctness/resource/performance gate.

_V105_CG_CONFIGS = {
    'qkv_gate': (2, 512),
    'o': (2, 32),
    'unpooler': (2, 256),
}
_V105_CG_SHAPES = {
    'qkv_gate': (2304, 8192),
    'o': (8192, 1024),
    'unpooler': (1024, 4096),
}
_V105_CG_INFO = {
    ('qkv_gate', 'base'): 0, ('qkv_gate', 'candidate'): 1,
    ('o', 'base'): 2, ('o', 'candidate'): 3,
    ('unpooler', 'base'): 4, ('unpooler', 'candidate'): 5,
}
_V105_REPLAY_OVERHEAD_US = 4.0
_V105_REQUIRED_CORE_RATIO = 0.97

_v105_launch_anchor = (
    '  if (K == 8192 && rows == 2 && nth == 256) '
    'cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>'
    '(a0, a1, xp, tp, pm, op, b4);'
)
if RGEMV_SRC.count(_v105_launch_anchor) != 1:
    raise RuntimeError('v105 cgemv launch marker mismatch')
RGEMV_SRC = RGEMV_SRC.replace(
    _v105_launch_anchor,
    '''  if (K == 1024 && rows == 2 && nth == 32) cgemv_kernel<2, 32, 1024><<<N / 2, 32 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 4096><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);''',
    1,
)

_v105_wrapper_anchor = '''void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {'''
if RGEMV_SRC.count(_v105_wrapper_anchor) != 1:
    raise RuntimeError('v105 cgemv wrapper marker mismatch')
RGEMV_SRC = RGEMV_SRC.replace(
    _v105_wrapper_anchor,
    r'''template <int ROWS, int NTH, int K>
int64_t cgemv_info_t() {
  auto kern = cgemv_kernel<ROWS, NTH, K>;
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kern, NTH + 32, 0));
  return (int64_t)attr.numRegs | ((int64_t)attr.localSizeBytes << 10) |
         ((int64_t)attr.sharedSizeBytes << 30) | ((int64_t)active << 50);
}

int64_t cgemv_info(int64_t which) {
  switch ((int)which) {
    case 0: return cgemv_info_t<2, 256, 8192>();
    case 1: return cgemv_info_t<2, 512, 8192>();
    case 2: return cgemv_info_t<2, 64, 1024>();
    case 3: return cgemv_info_t<2, 32, 1024>();
    case 4: return cgemv_info_t<2, 128, 4096>();
    case 5: return cgemv_info_t<2, 256, 4096>();
    default: TORCH_CHECK(false, "cgemv_info: bad instance");
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {''',
    1,
)
CPP_SRC += '\nint64_t cgemv_info(int64_t which);\n'


def _get_ext():
    """v105 production translation unit plus cgemv resource query."""
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace(
            "@@BODY@@", WG_HELPERS if sm90 else "")
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC +
                    _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v186_cg_{arch}" +
                 (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq",
                       "gqx", "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "rk"],
            extra_cuda_cflags=[
                "-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"
            ] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


def _v105_run_cgemv(x, layer, n, state, rows, nth):
    cg = state['cg'][layer]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().cgemv(cg['L'], cg['C'], x, cg['tab'], cg['pmax'], out,
                     cg['base4'], rows, nth)
    return out


def _v105_resource_info(layer, which):
    packed = int(_get_ext().cgemv_info(_V105_CG_INFO[(layer, which)]))
    return {
        'registers': packed & ((1 << 10) - 1),
        'local_bytes': (packed >> 10) & ((1 << 20) - 1),
        'static_shared_bytes': (packed >> 30) & ((1 << 20) - 1),
        'active_ctas_per_sm': packed >> 50,
    }


def _v105_validate_layer(weights, state, layer):
    w = weights[layer]
    n, k = _V105_CG_SHAPES[layer]
    rows_c, nth_c = _V105_CG_CONFIGS[layer]
    rows_b, nth_b = CG_CONFIGS[layer]
    seed = 105001 + tuple(_V105_CG_CONFIGS).index(layer)
    gen = torch.Generator(device=w.device).manual_seed(seed)
    x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                    device=w.device).to(torch.bfloat16)
    x[0, 0] = 0
    x[0, k - 1] = 16
    incumbent_cg = _v105_run_cgemv(x, layer, n, state, rows_b, nth_b)
    candidate = _v105_run_cgemv(x, layer, n, state, rows_c, nth_c)
    torch.cuda.synchronize(w.device)
    bitwise = bool(torch.equal(candidate, incumbent_cg))
    ref = (x.double() @ w[:n].double().t()).to(torch.bfloat16).double()
    rel = float(((candidate.double() - ref).norm() / ref.norm()).item())
    finite = bool(torch.isfinite(candidate).all())

    probe_exact = True
    one = torch.zeros((1, k), dtype=torch.bfloat16, device=w.device)
    for col in (0, k // 3, (2 * k) // 3, k - 1):
        one.zero_()
        one[0, col] = 1
        got = _v105_run_cgemv(one, layer, n, state, rows_c, nth_c)
        probe_exact = probe_exact and bool(torch.equal(got[0], w[:n, col]))
    torch.cuda.synchronize(w.device)
    return bitwise, probe_exact, finite, rel


_v105_base_prepare = prepare
_v105_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v105_base_prepare(weights)
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    state['_v105_cg'] = {}
    for layer, (rows_c, nth_c) in _V105_CG_CONFIGS.items():
        try:
            if layer not in state.get('cg', {}):
                continue
            n, k = _V105_CG_SHAPES[layer]
            resources_b = _v105_resource_info(layer, 'base')
            resources_c = _v105_resource_info(layer, 'candidate')
            bitwise, probe_exact, finite, rel = _v105_validate_layer(
                weights, state, layer)
            gen = torch.Generator(device=wq.device).manual_seed(
                105101 + tuple(_V105_CG_CONFIGS).index(layer))
            x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                            device=wq.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'm': 1, 'n': n, 'k': k, 'layer': layer}}
            cand_ms, base_ms = _v102_paired_graph_ms(
                lambda: _v105_run_cgemv(
                    x, layer, n, state, rows_c, nth_c),
                lambda: _v105_base_kernel_fn(data, state),
                wq.device, repeats=13)
            cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
            cand_core = max(cand_us - _V105_REPLAY_OVERHEAD_US, 1e-6)
            base_core = max(base_us - _V105_REPLAY_OVERHEAD_US, 1e-6)
            core_ratio = cand_core / base_core
            valid = (probe_exact and finite and rel < 5e-3 and
                     resources_c['local_bytes'] == 0 and
                     resources_c['active_ctas_per_sm'] >= 1)
            # v186: the exact QKV m=1 512-thread specialization is a
            # repeated small win whose old 3% prepare gate toggles on timing
            # noise.  Dispatch it deterministically after the existing
            # resource, finiteness, FP64, and one-hot coefficient validation.
            # O and unpooler retain the original conservative 3% gate.
            enable = (valid and (
                layer == 'qkv_gate' or
                (cand_us < base_us and
                 core_ratio <= _V105_REQUIRED_CORE_RATIO)))
            state['_v105_cg'][layer] = {'enabled': enable}
            print('v105 cgemv threads', layer, 'base', CG_CONFIGS[layer],
                  'candidate', (rows_c, nth_c), 'resources_base', resources_b,
                  'resources_candidate', resources_c, 'bitwise_base', bitwise,
                  'probe_exact', probe_exact, 'finite', finite, 'fp64_rel', rel,
                  'candidate_us', cand_us, 'incumbent_us', base_us,
                  'core_ratio', core_ratio, 'enabled', enable, flush=True)
        except Exception as exc:
            print('v105 cgemv threads unavailable', layer,
                  type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    gate = state.get('_v105_cg', {}).get(cfg['layer'])
    expected = _V105_CG_SHAPES.get(cfg['layer'])
    if (gate is not None and gate.get('enabled') and cfg['m'] == 1 and
            expected is not None and (cfg['n'], cfg['k']) == expected):
        rows, nth = _V105_CG_CONFIGS[cfg['layer']]
        x = data['x']
        cg = state['cg'][cfg['layer']]
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == cg['L'].device and x.data_ptr() % 16 == 0):
            return _v105_run_cgemv(
                x, cfg['layer'], cfg['n'], state, rows, nth)
    return _v105_base_kernel_fn(data, state)


# ---------------------------------------------------------------- v150: faithful RS128 producer/consumer pipeline.
# Apache-2.0 source: github.com/chenwainuo/okc @ 3d418e1.
# Upstream source SHA256: BB50079BE2ED74D65BB7109F1432FBC7A49BBE70913D247C254717CD0F27F712.
import time as _v150_clock
import hashlib as _v150_hashlib
import threading as _v150_threading

_V150_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;              // FL bit 20: per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// wf_rq128b: relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued (P4: 95 clk) -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// wf_dilp DILP (FL bit 12): the same 36 ALU ops, written as four independent 3-deep chains: every (hl, hh) pair is
// computed first (8 words, 20 ops, all independent), then the 16 PRMTs (all independent). rq_unit's per-dec form makes the
// compiler walk the four sub-blocks in order (each PRMT waits on its own LDS->LOP3->IADD chain); here the four chains'
// latencies overlap. Same values, same order of A[] writes.
__device__ __forceinline__ void rq_unit_ilp(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  const uint32_t hl0 = (q2.x & 0x87878787u) + Tb, hh0 = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  const uint32_t hl1 = (q2.y & 0x87878787u) + Tb, hh1 = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  const uint32_t hl2 = (q2.z & 0x87878787u) + Tb, hh2 = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  const uint32_t hl3 = (q2.w & 0x87878787u) + Tb, hh3 = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
  A[0] = rq_prmt(q0.x, hl0, 0x5140); A[1] = rq_prmt(q0.x, hl0, 0x7362);
  A[2] = rq_prmt(q0.y, hh0, 0x5140); A[3] = rq_prmt(q0.y, hh0, 0x7362);
  A[4] = rq_prmt(q0.z, hl1, 0x5140); A[5] = rq_prmt(q0.z, hl1, 0x7362);
  A[6] = rq_prmt(q0.w, hh1, 0x5140); A[7] = rq_prmt(q0.w, hh1, 0x7362);
  A[8] = rq_prmt(q1.x, hl2, 0x5140); A[9] = rq_prmt(q1.x, hl2, 0x7362);
  A[10] = rq_prmt(q1.y, hh2, 0x5140); A[11] = rq_prmt(q1.y, hh2, 0x7362);
  A[12] = rq_prmt(q1.z, hl3, 0x5140); A[13] = rq_prmt(q1.z, hl3, 0x7362);
  A[14] = rq_prmt(q1.w, hh3, 0x5140); A[15] = rq_prmt(q1.w, hh3, 0x7362);
}
// wf_rq128b WK: apply one patch-table word (bit 15 valid, 11:8 reg, 7 half, 6:0 xor7) to a decoded A fragment. The
// predicate is per-lane but the *instruction count* is warp-uniform, so the wgmma.fence no longer waits for a walker.
__device__ __forceinline__ void rq_patch(uint32_t (&A)[16], uint32_t w) {
  if (w & 0x8000u) {
    const int r = (int)((w >> 8) & 15);
    const uint32_t xr = (w & 0x7fu) << (8 + ((w >> 7) & 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
  }
}
// wf_rq128b FX2: the same walk with a fixed iteration count (warp-uniform instruction count, so the wgmma.fence of the
// warpgroup no longer waits for the warp that happens to walk the longest list). K must be >= the max hits per stage.
__device__ __forceinline__ void rq_fix2(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
#pragma unroll
  for (int it = 0; it < 2; it++) {
    if (nxs == (uint32_t)seq) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
      const bool me = lane == (int)((nx >> 12) & 31);
#pragma unroll
      for (int i = 0; i < 16; i++) if (me && i == r) A[i] ^= xr;
      ++ei;
      if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
      nx = __shfl_sync(0xffffffffu, ec, ei & 31);
      nxs = nx >> 17;
    }
  }
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
// wf_dilp WPF (FL bit 13): the same walk with the next entry's shfl issued before the current entry's 16 predicated XORs,
// so the shfl latency (the only dependent chain in the walk) hides under the apply instead of stalling the wgmma.fence.
__device__ __forceinline__ void rq_fixp(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    uint32_t nxn = __shfl_sync(0xffffffffu, ec, (ei + 1) & 31);
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) { ec = __ldg(el + ei + lane); nxn = __shfl_sync(0xffffffffu, ec, 0); }
    nx = nxn;
    nxs = nx >> 17;
  }
}
// FL bit 20: per-thread staged cursor.  E holds this thread's own escape entries sorted by stage (head = E.x),
// hs = E.x >> 17 kept pre-shifted so the common path is one compare + one branch; no shfl, no cross-lane dependency.
// The apply needs a runtime register index, so it is either 16 predicated XORs (if-converted -> 32 instrs EVERY stage,
// measured: loop body 2120 vs 1120 no-escape) or a real branch around them.  asm volatile cannot be if-converted, so
// rq_xor16 keeps the branch: common path = ISETP + BRA.  FL bit 22 = the __any_sync (warp-uniform branch) flavour.
__device__ __forceinline__ void rq_xor16(uint32_t (&A)[16], uint32_t nx) {
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16), r = (nx >> 8) & 15;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %16, 0;\n\t@p xor.b32 %0, %0, %17;\n\t"
      "setp.eq.u32 p, %16, 1;\n\t@p xor.b32 %1, %1, %17;\n\t"
      "setp.eq.u32 p, %16, 2;\n\t@p xor.b32 %2, %2, %17;\n\t"
      "setp.eq.u32 p, %16, 3;\n\t@p xor.b32 %3, %3, %17;\n\t"
      "setp.eq.u32 p, %16, 4;\n\t@p xor.b32 %4, %4, %17;\n\t"
      "setp.eq.u32 p, %16, 5;\n\t@p xor.b32 %5, %5, %17;\n\t"
      "setp.eq.u32 p, %16, 6;\n\t@p xor.b32 %6, %6, %17;\n\t"
      "setp.eq.u32 p, %16, 7;\n\t@p xor.b32 %7, %7, %17;\n\t"
      "setp.eq.u32 p, %16, 8;\n\t@p xor.b32 %8, %8, %17;\n\t"
      "setp.eq.u32 p, %16, 9;\n\t@p xor.b32 %9, %9, %17;\n\t"
      "setp.eq.u32 p, %16, 10;\n\t@p xor.b32 %10, %10, %17;\n\t"
      "setp.eq.u32 p, %16, 11;\n\t@p xor.b32 %11, %11, %17;\n\t"
      "setp.eq.u32 p, %16, 12;\n\t@p xor.b32 %12, %12, %17;\n\t"
      "setp.eq.u32 p, %16, 13;\n\t@p xor.b32 %13, %13, %17;\n\t"
      "setp.eq.u32 p, %16, 14;\n\t@p xor.b32 %14, %14, %17;\n\t"
      "setp.eq.u32 p, %16, 15;\n\t@p xor.b32 %15, %15, %17;\n\t}"
      : "+r"(A[0]), "+r"(A[1]), "+r"(A[2]), "+r"(A[3]), "+r"(A[4]), "+r"(A[5]), "+r"(A[6]), "+r"(A[7]),
        "+r"(A[8]), "+r"(A[9]), "+r"(A[10]), "+r"(A[11]), "+r"(A[12]), "+r"(A[13]), "+r"(A[14]), "+r"(A[15])
      : "r"(r), "r"(xr));
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void rq_xor16s(uint32_t (&A)[16], uint32_t nx) {  // if-convertible 16-way select
  const int r = (int)((nx >> 8) & 15);
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
  for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
}
__device__ __forceinline__ void rq_fixc(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 20: real branch
  while (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
}
__device__ __forceinline__ void rq_fixv(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 22: warp-uniform
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16s(A, E.x); rq_adv(E, hs); }
  }
}
// bit 23: warp-uniform branch (__any_sync) around the asm apply -> the apply is neither if-converted nor hoisted out of
// the loop (micro: loop body 107 vs 101 no-escape, 115 vote+select, 199 asm alone), so wgmma.fence has nothing to wait for
__device__ __forceinline__ void rq_fixw(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
  }
}
// FL bit 25 (wf_escur3 H2b, ported): patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140); A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140); A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140); A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140); A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140); A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140); A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140); A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140); A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
template <> __device__ __forceinline__ void rq_wgmma<16>(float (&D)[8], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgmma<32>(float (&D)[16], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgmma<128>(float (&D)[64], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgmma<256>(float (&D)[128], const uint32_t* a, uint64_t desc, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63]), "+f"(D[64]), "+f"(D[65]), "+f"(D[66]), "+f"(D[67]), "+f"(D[68]), "+f"(D[69]), "+f"(D[70]), "+f"(D[71]), "+f"(D[72]), "+f"(D[73]), "+f"(D[74]), "+f"(D[75]), "+f"(D[76]), "+f"(D[77]), "+f"(D[78]), "+f"(D[79]), "+f"(D[80]), "+f"(D[81]), "+f"(D[82]), "+f"(D[83]), "+f"(D[84]), "+f"(D[85]), "+f"(D[86]), "+f"(D[87]), "+f"(D[88]), "+f"(D[89]), "+f"(D[90]), "+f"(D[91]), "+f"(D[92]), "+f"(D[93]), "+f"(D[94]), "+f"(D[95]), "+f"(D[96]), "+f"(D[97]), "+f"(D[98]), "+f"(D[99]), "+f"(D[100]), "+f"(D[101]), "+f"(D[102]), "+f"(D[103]), "+f"(D[104]), "+f"(D[105]), "+f"(D[106]), "+f"(D[107]), "+f"(D[108]), "+f"(D[109]), "+f"(D[110]), "+f"(D[111]), "+f"(D[112]), "+f"(D[113]), "+f"(D[114]), "+f"(D[115]), "+f"(D[116]), "+f"(D[117]), "+f"(D[118]), "+f"(D[119]), "+f"(D[120]), "+f"(D[121]), "+f"(D[122]), "+f"(D[123]), "+f"(D[124]), "+f"(D[125]), "+f"(D[126]), "+f"(D[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgss<128>(float (&D)[64], uint64_t da, uint64_t db, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63])
    : "l"(da), "l"(db), "r"(scd));
}
template <> __device__ __forceinline__ void rq_wgss<256>(float (&D)[128], uint64_t da, uint64_t db, int scd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]), "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]), "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]), "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]), "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]), "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]), "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]), "+f"(D[62]), "+f"(D[63]), "+f"(D[64]), "+f"(D[65]), "+f"(D[66]), "+f"(D[67]), "+f"(D[68]), "+f"(D[69]), "+f"(D[70]), "+f"(D[71]), "+f"(D[72]), "+f"(D[73]), "+f"(D[74]), "+f"(D[75]), "+f"(D[76]), "+f"(D[77]), "+f"(D[78]), "+f"(D[79]), "+f"(D[80]), "+f"(D[81]), "+f"(D[82]), "+f"(D[83]), "+f"(D[84]), "+f"(D[85]), "+f"(D[86]), "+f"(D[87]), "+f"(D[88]), "+f"(D[89]), "+f"(D[90]), "+f"(D[91]), "+f"(D[92]), "+f"(D[93]), "+f"(D[94]), "+f"(D[95]), "+f"(D[96]), "+f"(D[97]), "+f"(D[98]), "+f"(D[99]), "+f"(D[100]), "+f"(D[101]), "+f"(D[102]), "+f"(D[103]), "+f"(D[104]), "+f"(D[105]), "+f"(D[106]), "+f"(D[107]), "+f"(D[108]), "+f"(D[109]), "+f"(D[110]), "+f"(D[111]), "+f"(D[112]), "+f"(D[113]), "+f"(D[114]), "+f"(D[115]), "+f"(D[116]), "+f"(D[117]), "+f"(D[118]), "+f"(D[119]), "+f"(D[120]), "+f"(D[121]), "+f"(D[122]), "+f"(D[123]), "+f"(D[124]), "+f"(D[125]), "+f"(D[126]), "+f"(D[127])
    : "l"(da), "l"(db), "r"(scd));
}


template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool MRG = (FL & 64) != 0;    // one empty barrier per slot: the x release is folded into the W release
  constexpr bool RLAX = (FL & 128) != 0;  // relaxed (non-release) empty arrive
  constexpr bool NOE = (FL & 512) != 0;   // PROBE: no escape walk (wrong output, times the floor)
  constexpr bool FX2 = (FL & 16384) != 0;  // fixed-iteration uniform escape loop (bit 14: bit 10 is pw47's H2)
  constexpr bool WK = (FL & 32768) != 0;   // patch-table walker (warp PW+3) instead of the per-warp escape list walk (bit 15: bit 11 is pw47's F2)
  constexpr bool DILP = (FL & 4096) != 0; // wf_dilp: 4 independent decode chains (rq_unit_ilp)
  constexpr bool WPF = (FL & 8192) != 0;  // wf_dilp: escape walk with the next-entry shfl prefetched
  static_assert(!WK || (PRB == 0 && MS == 1), "WK: full kernel, MS == 1");  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && GS == MS && (ST >= 5 && ST <= 8) && PRB == 0 && WM <= 1 && INS == 0 && (FL == 65538 || FL == 65539 || FL == 65570 || FL == 65571))
                || (NT == 256 && GS == MS && ST == 5 && PRB == 0 && WM == 1 && INS == 0 && FL == 65603), "EH backport approved M128/ST5-8/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // wf_rq32b: 2 CTAs per SM. FL bit 10 (H2): 256-thread CTA = 4 math warps + W lane + x lane + 2 zeroing warps, HALF
  // (64 W rows) only, launch cap 128 regs, math 216 / producer 40. FL bit 11 (F2): 384 threads, full 128 rows, launch
  // cap 84 regs, math 104 / producer 40 (KS must double so the grid still fills 2 x 132 SMs).
  constexpr bool H2 = (FL & 1024) != 0, F2 = (FL & 2048) != 0;
  static_assert(!H2 || HALF, "H2 needs HALF (64 W rows, one math warpgroup)");
  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  static_assert(MS == 1 || (GS == MS && !HALF), "MS > 1: GS == MS and no HALF");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
  uint32_t* const ptabp = (uint32_t*)(smem + (sbase - sraw) + ptab);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  constexpr bool CUR = (FL & 1048576) != 0, NOESC = (FL & 2097152) != 0, VOTE = (FL & 4194304) != 0,
                 VASM = (FL & 8388608) != 0;
  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  // wf_rq32 FL bit 8 (ESPLIT): the entry x tiles are issued by the x lane (warp 9) in parallel with the W lane's bulks
  // (one bar.sync 6,64 after the init fence); bit 9 (EFENCE): the epilogue's fence.proxy.async.global is dropped (the
  // zeroing warp already fenced its own stores before publishing zflag).  Both measured within noise (batch 3):
  // ESPLIT -0.06..+0.11 us, EFENCE -0.22..+0.05 us on qkv m=32/16 -> no RQV entry uses them (dead code, kept for the
  // record; FL 35 is unchanged).
  constexpr bool ESPLIT = (FL & 256) != 0, EFENCE = (FL & 512) != 0;
  uint32_t ec = (warp < NW && !CUR) ? __ldg(el + lane) : 0u;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (CUR && warp < NW)  // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  if (ESPLIT && EARLY) {
    if (warp == PW && lane == 0) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == PW || (warp == PW + 1 && XON)) asm volatile("bar.sync 6, 64;\n" ::: "memory");
    if (warp == PW && lane == 0) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t px = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  } else if (tid == (EARLY ? PW * 32 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < MS * ST && s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS, slot = (s / MS) % ST;
        rq_expect(full + 8 * slot, n * WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * slot, n * XB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, px);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
      }
    }
  }
  if (tid == (PW + 2) * 32 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t ec0 = 0xffffffffu;
  if (WK && warp == PW + 3 && lane < 8) ec0 = __ldg(p.esc + (size_t)((ks * p.NB + nb) * 8 + (lane & 7)) * p.ecap);
  auto walk = [&]() {  // WK patch-table walker (warp PW+3): lane l walks math warp l's escape list and publishes
    // (hdr, 2 words per lane) into the slot's table before arriving on the slot's full barrier
    const int lw = lane & 7;
    const uint32_t* myel = p.esc + (size_t)((ks * p.NB + nb) * 8 + lw) * p.ecap;
    uint32_t* const tb0 = ptabp + lw * C::PTW;
    int ei = 0, slot = 0, r = 0;
    uint32_t ec = ec0, u0 = 0, u1 = 0;
    for (int s = 0; s < KCS; s++) {
      if (s >= ST) rq_wait((MRG ? wempty : xempty) + 8 * slot, (r & 1) ^ 1);
      uint32_t* const tb = tb0 + slot * 8 * C::PTW;
      if (lane < 8) {  // clear the words this slot's previous fill set (the consumer is done with them: xempty)
        uint32_t q0 = tb[1], q1 = tb[2];
        while (q0) { const int l = __ffs((int)q0) - 1; tb[3 + l * 2] = 0u; q0 &= q0 - 1; }
        while (q1) { const int l = __ffs((int)q1) - 1; tb[4 + l * 2] = 0u; q1 &= q1 - 1; }
        tb[0] = 0u; tb[1] = 0u; tb[2] = 0u; u0 = u1 = 0u;
      }
      while (lane < 8 && (ec >> 17) == (uint32_t)s) {
        const int lf = (int)((ec >> 12) & 31), idx = (int)((u0 >> lf) & 1);
        tb[3 + lf * 2 + idx] = 0x8000u | (((ec >> 8) & 15u) << 8) | (((ec >> 7) & 1u) << 7) | (ec & 0x7fu);
        if (idx) u1 |= 1u << lf; else u0 |= 1u << lf;
        ++ei;
        ec = __ldg(myel + ei);
      }
      if (lane < 8) { tb[1] = u0; tb[2] = u1; tb[0] = (u0 | u1) ? 1u : 0u; }
      __syncwarp();
      if (lane == 0) rq_arrive(full + 8 * slot);
      if (++slot == ST) { slot = 0; r++; }
    }
  };
  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * WB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : (MRG ? wempty : xempty)) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * XB);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= PW + 2) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
      if (WK && warp == PW + 3) walk();
    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : (s / MS) % ST;
        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES && !MRG) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) {
          const uint32_t b = (MRG ? wempty : xempty) + 8 * (((GS * G + j) / MS) % ST);
          if (RLAX) rq_arrive_r(b); else rq_arrive(b);
        }
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          if constexpr (CURP) {
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
          } else {
          const uint32_t* const tb = ptabp + ((((GS * G + j) / MS) % ST) * 8 + (warp & 7)) * C::PTW;
          const uint32_t ph = WK ? tb[0] : 0u, pw0 = WK ? tb[3 + lane * 2] : 0u, pw1 = WK ? tb[4 + lane * 2] : 0u;
          if (DILP) rq_unit_ilp(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          else rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          if (WK) {
            if (__builtin_expect(ph != 0u, 0)) { rq_patch(A[j], pw0); rq_patch(A[j], pw1); }
          } else if constexpr (CUR && VASM) rq_fixw(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR && VOTE) rq_fixv(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR) rq_fixc(A[j], GS * G + j, E4, hs);
          else if (NOE) { } else if (FX2) rq_fix2(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else if (WPF) rq_fixp(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          }
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : ((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        if (!EFENCE) asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL, MS>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (info == 4) {  // wf_rq32b: CTAs per SM at this launch geometry (regs + smem)
      int nb = 0;
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, (FL & 1024) ? 256 : 384, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -1; }
      return nb;
    }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    if (esc2.numel() && esc2.numel() < nl * 128) return -14;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.esc2 = esc2.numel() ? (const uint32_t*)esc2.data_ptr() : nullptr;
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= 2 * KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
  }
  return -2;
}
"""
_V150_VARIANT = ('d2s5hzwP', 128, 5, 1, 0, 0, 0, 42999939)
_V150_CPP = "int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);"


def _v150_cuda_source():
    return _V150_CUDA_SOURCE


_v150_module = [None]


def _v150_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v150_rs128_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
        _v150_module[0] = load_inline(
            tag,
            cpp_sources=_V150_CPP,
            cuda_sources=src,
            functions=["rqrun"],
            extra_cuda_cflags=[
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3", "-DNDEBUG", "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _v150_module[0]


_V150_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))

_V150_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))

def _v150_c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    T, KC = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_V150_OS[dv], device=dev)) | (sg.long() << torch.tensor(_V150_SS[dv], device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
    if not torch.equal(_v150_unc12(data, tab, esc, dv), b):
        return None
    return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc

def _v150_unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    T, KC = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = (C[..., None] >> torch.tensor(_V150_OS[dv], device=dev)) & 7
    sgn = (C[..., None] >> torch.tensor(_V150_SS[dv], device=dev)) & 1
    tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
    lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
    val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        t, c, j, l, vv, x = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()

def _v150_pack_nat(w):
    """pw10 _v150_pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()

def _v150_pack(w):
    """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
    N, K = w.shape
    c = _v150_c12(_v150_pack_nat(w), 1)
    if c is None or N % 128:
        return None
    data, tab, esc = c
    T, KC = data.shape[:2]
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    return lay, tab, esc

def _v150_unpack(lay, tab, esc, N, K):
    """independent check of the re-blocking: unpack [NB][KC][8][1536] with _v150_unc12 and undo _v150_pack_nat -> bf16 [N, K]"""
    NB, KC = lay.shape[:2]
    data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
    v = _v150_unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
    W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
    return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)

def _v150_esc(esc6, NB, KC, KS, dev):
    """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
    G = KS * NB * 8
    if len(esc6):
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        gw = (ks * NB + nb) * 8 + wt
        o = torch.argsort(gw * 65536 + u)
        gw, ent = gw[o], ent[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
    else:
        cap = 32
    arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
    if len(esc6):
        arr[gw * cap + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)

def _v150_esc2(esc6, NB, KC, KS, dev, CAPT=4):
    """FL bit 20 lists: one fixed slot of CAPT entries per (ks, nb, warp, lane), sorted by stage, sentinel-terminated
    -> int32 [KS*NB*8*32*CAPT].  Returns (arr, max entries in any slot); arr is None if a slot overflows CAPT."""
    G = KS * NB * 8
    if not len(esc6):
        return torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev), 0
    tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
    nb, wt = tile // 8, tile % 8
    s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
    ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
    u = ch - s0[ks]
    ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
    slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
    o = torch.argsort(slot * 65536 + u)
    slot, ent = slot[o], ent[o]
    rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
    mx = int(rank.max()) + 1
    if mx > CAPT:
        return None, mx
    arr = torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
    arr[slot * CAPT + rank] = ent
    return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx

_V150_TARGETS = ("qkv_gate", "pooler", "unpooler")
_V150_KS = {"qkv_gate": 7, "pooler": 4, "unpooler": 2}
_V150_TOL = {2: 5e-3, 4: 6e-3, 7: 8e-3}
_V150_ROUTE_RATIO = 0.97
_V150_SCORE_GATE = 0.009
_V150_REPLAY_US = 4.0
_V150_FLUSH = [None, None]


def _v150_start_build():
    status = []

    def build():
        try:
            _v150_build()
            status.append("ok")
        except Exception as exc:
            status.append("failed: " + (str(exc).splitlines()[-1][:240] if str(exc) else type(exc).__name__))

    thread = _v150_threading.Thread(target=build, daemon=True)
    thread.start()
    return thread, status


def _v150_call(state, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        0, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v150 rqrun error %d" % rc)
    return out


def _v150_info(state, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        0, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v150_validate(state, plan, weight, layer):
    m, n, k, ks = 128, weight.shape[0], weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v150_call(state, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    if not exact or int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        print("v150 rejected", layer, "one-hot/counter", exact, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(1500 + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v150_call(state, plan, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL[ks]
    ok = bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol
    if int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        ok = False
    print(
        "v150 exact", layer, "one_hot", exact, "fp64", rel, worst,
        "regs", _v150_info(state, plan, 1), "local", _v150_info(state, plan, 2),
        "smem", _v150_info(state, plan, 3), "esc2_max", plan["esc2_max"],
        "valid", ok, flush=True,
    )
    return ok


def _v150_time(fn, flush, repeats=11):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fn()
    if _V150_FLUSH[0] is not flush:
        _V150_FLUSH[0] = flush
        _V150_FLUSH[1] = torch.empty(1024, dtype=torch.float32, device=flush.device)
    samples = []
    for _ in range(repeats):
        torch.sum(flush, dim=0, out=_V150_FLUSH[1])
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _v150_pair(candidate, incumbent, flush, repeats=11, rounds=3):
    ratios, cand, inc = [], [], []
    for round_index in range(rounds):
        if round_index & 1:
            b = _v150_time(incumbent, flush, repeats)
            a = _v150_time(candidate, flush, repeats)
        else:
            a = _v150_time(candidate, flush, repeats)
            b = _v150_time(incumbent, flush, repeats)
        ratios.append(
            max(a - _V150_REPLAY_US, 0.05) /
            max(b - _V150_REPLAY_US, 0.05)
        )
        cand.append(a); inc.append(b)
    ratios.sort(); cand.sort(); inc.sort()
    middle = len(ratios) // 2
    return ratios[middle], cand[middle], inc[middle]


_v105_prepare = prepare
_v105_kernel_fn = kernel_fn


def prepare(weights):
    started = _v150_clock.time()
    dev = weights["pooler"].device
    props = torch.cuda.get_device_properties(dev)
    eligible = props.major == 9 and "H200" in props.name
    build_thread = build_status = None
    if eligible:
        build_thread, build_status = _v150_start_build()

    state = _v105_prepare(weights)
    state["v150_enabled"] = set()
    state["v150_projected_gain"] = 0.0
    if not eligible:
        return state

    try:
        build_thread.join(timeout=max(1.0, 220.0 - (_v150_clock.time() - started)))
        if not build_status or build_status[0] != "ok":
            print("v150 unavailable build", build_status or "not finished", flush=True)
            return state
        vstate = {
            "ext": _v150_build(),
            "prof": torch.empty(0, dtype=torch.int64, device=dev),
            "sink": torch.zeros(4, dtype=torch.int32, device=dev),
            "plans": {},
        }
        state["v150"] = vstate

        for layer in _V150_TARGETS:
            weight = weights[layer]
            packed = _v150_pack(weight)
            if packed is None:
                print("v150 rejected", layer, "pack unavailable", flush=True)
                continue
            lay, tab, esc6 = packed
            if not torch.equal(
                _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                weight.view(torch.int16).int() & 0xffff,
            ):
                print("v150 rejected", layer, "pack mismatch", flush=True)
                continue
            ks = _V150_KS[layer]
            esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
            esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
            if esc2 is None:
                print("v150 rejected", layer, "esc2 overflow", esc2_max, flush=True)
                continue
            plan = {
                "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                "esc2_max": esc2_max, "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                "KS": ks, "n": weight.shape[0],
            }
            vstate["plans"][layer] = plan
            if not _v150_validate(vstate, plan, weight, layer):
                del vstate["plans"][layer]

        if not vstate["plans"]:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        qualified = set()
        route_times = {}
        inc_total = cand_total = 0.0
        gen = torch.Generator(device=dev).manual_seed(150128)
        for layer in _V150_TARGETS:
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 128, "n": weight.shape[0], "k": weight.shape[1]}
            plan = vstate["plans"].get(layer)
            if plan is None:
                incumbent_us = _v150_time(lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state), flush)
                inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
                inc_total += inc_core; cand_total += inc_core
                continue
            out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16, device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda x=x, out=out, plan=plan: _v150_call(vstate, plan, x, out),
                lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            keep = candidate_us < incumbent_us and ratio <= _V150_ROUTE_RATIO
            if keep:
                qualified.add((128, layer))
            route_times[layer] = (candidate_us, incumbent_us, ratio, keep)
            inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
            cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
            inc_total += inc_core; cand_total += cand_core
            print(
                "v150 rs128", layer, "candidate", candidate_us,
                "incumbent", incumbent_us, "core_ratio", ratio,
                "qualified", keep, flush=True,
            )

        for layer in ("o", "router"):
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 128, "n": weight.shape[0], "k": weight.shape[1]}
            unchanged_us = _v150_time(
                lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            unchanged_core = max(unchanged_us - _V150_REPLAY_US, 0.05)
            inc_total += unchanged_core; cand_total += unchanged_core

        load_ratio = inc_total / cand_total
        projected = load_ratio ** (1.0 / 6.0) - 1.0
        state["v150_projected_gain"] = projected
        if projected >= _V150_SCORE_GATE:
            state["v150_enabled"] = qualified
        print(
            "v150 load128 incumbent", inc_total, "candidate", cand_total,
            "ratio", load_ratio, "projected_score_gain", projected,
            "enabled", sorted(state["v150_enabled"]), flush=True,
        )
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v150 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v150_enabled", ()):
        plan = state["v150"]["plans"].get(cfg["layer"])
        if plan is not None:
            return _v150_call(state["v150"], plan, data["x"])
    return _v105_kernel_fn(data, state)


# ---------------------------------------------------------------- v158: exact RS16/RS32 guarded tournament.
_V158_ROUTES = {
    (32, "qkv_gate"): 1,
}
_V158_KS = {"qkv_gate": 7, "pooler": 4, "unpooler": 2, "o": 2}
_V158_ROUTE_RATIO = 0.97
_V158_SCORE_GATE = 0.0


def _v158_call(vstate, plan, x, out=None):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = vstate["ext"].rqrun(
        plan["variant"], plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], 0,
    )
    if rc != 0:
        raise RuntimeError("v158 rqrun error %d" % rc)
    return out


def _v158_info(vstate, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(vstate["ext"].rqrun(
        plan["variant"], empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], kind,
    ))


def _v158_validate(vstate, plan, weight, layer, m):
    n, k, ks = weight.shape[0], weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v158_call(vstate, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want)
    if len(cols) < m:
        exact = exact and bool((out[len(cols)] == 0).all())

    gen = torch.Generator(device=weight.device).manual_seed(158000 + m + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v158_call(vstate, plan, x)
    torch.cuda.synchronize()
    ref = x[:min(8, m)].double() @ weight.double().T
    diff = out[:min(8, m)].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL[ks]
    counters_ok = int(plan["cnt"].abs().sum()) == 0
    if not counters_ok:
        plan["cnt"].zero_()
    ok = exact and bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol and counters_ok
    print(
        "v158 exact", layer, "m", m, "variant", plan["variant"],
        "one_hot", exact, "fp64", rel, worst,
        "regs", _v158_info(vstate, plan, 1), "local", _v158_info(vstate, plan, 2),
        "smem", _v158_info(vstate, plan, 3), "occupancy", _v158_info(vstate, plan, 4),
        "valid", ok, flush=True,
    )
    return ok


_v150_prepare_with_rs128 = prepare
_v150_kernel_with_rs128 = kernel_fn


def prepare(weights):
    state = _v150_prepare_with_rs128(weights)
    state["v158_enabled"] = set()
    state["v158_projected_gain"] = 0.0
    vstate = state.get("v150")
    if vstate is None or not state.get("v150_enabled"):
        return state

    dev = weights["pooler"].device
    try:
        plans = {}
        for (m, layer), variant in _V158_ROUTES.items():
            source_plan = vstate["plans"].get(layer)
            if source_plan is None:
                weight = weights[layer]
                packed = _v150_pack(weight)
                if packed is None:
                    continue
                lay, tab, esc6 = packed
                if not torch.equal(
                    _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                    weight.view(torch.int16).int() & 0xffff,
                ):
                    continue
                ks = _V158_KS[layer]
                esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
                esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
                if esc2 is None:
                    continue
                source_plan = {
                    "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                    "esc2_max": esc2_max,
                    "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                    "KS": ks, "n": weight.shape[0],
                }
            plan = dict(source_plan)
            plan["variant"] = variant
            if _v158_validate(vstate, plan, weights[layer], layer, m):
                plans[(m, layer)] = plan
        state["v158_plans"] = plans
        if not plans:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(158032)
        qualified = set()
        ratios_by_m = {}
        for m in (16, 32):
            inc_total = cand_total = 0.0
            for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
                weight = weights[layer]
                x = torch.randn((m, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
                cfg = {"layer": layer, "m": m, "n": weight.shape[0], "k": weight.shape[1]}
                plan = plans.get((m, layer))
                if plan is None:
                    unchanged = _v150_time(
                        lambda x=x, cfg=cfg: _v150_kernel_with_rs128({"x": x, "config": cfg}, state), flush,
                    )
                    core = max(unchanged - _V150_REPLAY_US, 0.05)
                    inc_total += core; cand_total += core
                    continue
                out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=dev)
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda x=x, out=out, plan=plan: _v158_call(vstate, plan, x, out),
                    lambda x=x, cfg=cfg: _v150_kernel_with_rs128({"x": x, "config": cfg}, state),
                    flush,
                )
                keep = candidate_us < incumbent_us and ratio <= _V158_ROUTE_RATIO
                if keep:
                    qualified.add((m, layer))
                inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
                cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
                inc_total += inc_core; cand_total += cand_core
                print(
                    "v158 rs-small", layer, "m", m, "variant", plan["variant"],
                    "candidate", candidate_us, "incumbent", incumbent_us,
                    "core_ratio", ratio, "qualified", keep, flush=True,
                )
            ratios_by_m[m] = inc_total / cand_total
            print("v158 load", m, "incumbent", inc_total, "candidate", cand_total,
                  "ratio", ratios_by_m[m], flush=True)

        projected = (ratios_by_m[16] * ratios_by_m[32]) ** (1.0 / 6.0) - 1.0
        state["v158_projected_gain"] = projected
        if projected >= _V158_SCORE_GATE:
            state["v158_enabled"] = qualified
        print(
            "v158 projected_score_gain", projected,
            "enabled", sorted(state["v158_enabled"]), flush=True,
        )
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v158 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v158_enabled", ()):
        plan = state.get("v158_plans", {}).get(key)
        if plan is not None:
            return _v158_call(state["v150"], plan, data["x"])
    return _v150_kernel_with_rs128(data, state)


# v126: slot-owned exact-C12 correction scheduling for every dense SK2 route.
# The protected kernel launches NESC correction warps and makes all of them
# observe and acknowledge every ring stage, although only st % NESC performs
# corrections.  Here a stage has exactly one correction-warp participant and
# every empty barrier expects NW consumers plus that one owner.  The NESC
# correction buffers and stage assignment are unchanged, so correction work
# stays parallel and FP32 accumulation order is identical.
import hashlib as _v126_hashlib
import math as _v126_math


def _v126_patch_once(src, old, new, count=1):
    if src.count(old) != count:
        raise RuntimeError(
            f'v126 source marker mismatch: expected {count}, got {src.count(old)}')
    return src.replace(old, new)


def _v126_cut_once(src, begin, end, replacement):
    if src.count(begin) != 1:
        raise RuntimeError('v126 source range marker mismatch')
    i = src.index(begin)
    j = src.index(end, i + len(begin))
    return src[:i] + replacement + src[j + len(end):]


def _v126_info_source():
    lines = [r'''
#include <vector>
std::vector<int64_t> sk2s_info(int64_t cfg, int64_t tokr, int64_t pmax) {
  cudaFuncAttributes attr;
  int active = 0;
''']
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if layer == 'router':
            continue
        for tk in (8, 16, 32):
            if SK2_PICK.get((layer, tk)) != name:
                continue
            _, nstage = _sk2_geom(cfg, tk)
            mt = 2 if tk == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            nw = NWR * WSK
            nesc = 4 if nstage >= 4 else nstage
            ups = WSK * UPW
            wby = nw * UPW * NT * 768
            xby = ups * tk * 128
            sst = (xby + wby + 1023) & ~1023
            toks = 16 * mt
            rp = R + ((40 - (R % 32)) % 32)
            cp = R + 1
            csz = (toks * cp + 3) & ~3
            peer = ((CL - 1) * WSK * toks * rp * 4) if CL > 1 else 0
            smem_fixed = (1024 + nstage * sst + 16 * nstage + 32 +
                          nesc * csz * 4 + peer + 128)
            pk = 'true' if tk == 8 else 'false'
            lines.append(f'''  if (cfg == {ci} && tokr == {tk}) {{
    auto kern = sk2s_kernel<{mt}, {tk}, {NT}, {NWR}, {WSK}, {UPW},
                             {nstage}, true, false, {nacc}, {CL},
                             __nv_bfloat16, {pk}>;
    const size_t smem = (size_t){smem_fixed} + (size_t)pmax * 8;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, kern, {(nw + 1 + nesc) * 32}, smem));
    return {{(int64_t)attr.numRegs, (int64_t)attr.localSizeBytes,
             (int64_t)attr.sharedSizeBytes, (int64_t)active,
             (int64_t)smem}};
  }}
''')
    lines.append(r'''  TORCH_CHECK(false, "sk2s_info: bad instance");
}
''')
    return ''.join(lines)


def _v126_sk2s_src(sm90):
    if not sm90:
        return r'''
#include <vector>
void sk2s(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t,
          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
          int64_t, at::Tensor) { TORCH_CHECK(false, "sk2s needs sm_90"); }
std::vector<int64_t> sk2s_info(int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "sk2s_info needs sm_90");
}
'''
    src = _sk2_src(True)
    src = _v126_patch_once(
        src, 'NW + (COMP ? NESC : 0)', 'NW + (COMP ? 1 : 0)', 4)
    src = _v126_cut_once(
        src,
        '  if (warp >= NW + 1) {',
        '  // ---------------- consumers',
        r'''  if (warp >= NW + 1) {
    // One correction warp owns each stage.  It alone waits on and returns
    // that stage's slot, so the producer cannot reuse the slot early; no
    // non-owner needs to poll an unrelated full barrier.
    const int ew = warp - (NW + 1);
    SK2_DLY_PICK()
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = ew; st < nst; st += NESC) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kbeg) ++pe;
        const int slot = st % NSTAGE;
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 +
                (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) +
                (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] +=
                esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        SK2_DELAY_HOOK(st);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1)
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers''')
    # Keep the production source and candidate source in one extension while
    # giving every candidate symbol and debug macro a distinct namespace.
    src = src.replace('SK2', 'SK2S').replace('sk2', 'sk2s')
    return src + _v126_info_source()


CPP_SRC += r'''
#include <vector>
#include <pybind11/stl.h>
void sk2s(torch::Tensor Ws, torch::Tensor x, torch::Tensor out,
          torch::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
          int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg,
          torch::Tensor dbgt);
std::vector<int64_t> sk2s_info(int64_t cfg, int64_t tokr, int64_t pmax);
'''

_v126_ext = None


def _get_ext():
    global _v126_ext
    if _v126_ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f'{major}{minor}a' if sm90 else f'{major}{minor}'
        common = CUDA_SRC.replace('@@MBAR@@', MBAR_HELPERS).replace(
            '@@BODY@@', WG_HELPERS if sm90 else '')
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ''
        sk2s = _v126_sk2s_src(sm90)
        sk2l = _v134_sk2_l2_src(sm90)
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    sk2s + sk2l + _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC +
                    _rk_src(sm90))
        digest = _v126_hashlib.sha256(sk2s.encode()).hexdigest()[:16]
        _v126_ext = load_inline(
            name=f'okc_bf16proj_v126_slotowner_{arch}_{digest}',
            cpp_sources='#include <torch/extension.h>\n' + CPP_SRC,
            cuda_sources=cuda_src,
            functions=['skinny_tma_a', 'gemm90', 'wide', 'sk2', 'sk2s',
                       'sk2s_info', 'sk2_l2', 'sk2_l2_info',
                       'gq', 'gqx', 'rgemv', 'sgemv', 'cgemv',
                       'cgemv_info', 'rk'],
            extra_cuda_cflags=['-O3', '-lineinfo',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}'],
            no_implicit_headers=True, verbose=False)
    return _v126_ext


def _v126_run(x, layer, m, n, k, state):
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, _ = state['sk2'][name]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().sk2s(stream, x, out, tab, pmax, b4, m, n, k, ci,
                    _sk2_tokr(m), 0, state['dummy'])
    return out


def _v126_resources(layer, m, state):
    name = _sk2_name(layer, m)
    ci, _, _, pmax, _, _ = state['sk2'][name]
    vals = list(_get_ext().sk2s_info(ci, _sk2_tokr(m), pmax))
    return {'registers': vals[0], 'local_bytes': vals[1],
            'static_shared_bytes': vals[2], 'active_ctas_per_sm': vals[3],
            'dynamic_shared_bytes': vals[4]}


def _v126_validate(weights, state, layer, m):
    w = weights[layer]
    n = 1024 if layer == 'unpooler' and m <= 8 else w.shape[0]
    k = w.shape[1]
    gen = torch.Generator(device=w.device).manual_seed(
        126000 + m * 17 + sum(map(ord, layer)))
    x = torch.randn((m, k), dtype=torch.bfloat16, device=w.device,
                    generator=gen)
    candidate = _v126_run(x, layer, m, n, k, state)
    sk2_base = _run_sk2(x, layer, m, n, k, state)
    bitwise = bool(torch.equal(candidate, sk2_base))
    finite = bool(torch.isfinite(candidate).all())
    rows = torch.linspace(0, n - 1, steps=min(32, n), device=w.device).long()
    ref = x.double() @ w[rows].double().t()
    got = candidate[:, rows].double()
    rel_fp64 = float((got - ref).norm() / ref.norm().clamp_min(1e-20))
    cols = list(dict.fromkeys([0, k // 5, k // 3, k // 2, k - 1]))
    probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
    for i, col in enumerate(cols[:m]):
        probe[i, col] = 1
    probed = _v126_run(probe, layer, m, n, k, state)
    probe_exact = all(torch.equal(probed[i], w[:n, col])
                      for i, col in enumerate(cols[:m]))
    return bitwise, finite, rel_fp64, probe_exact


_v126_base_prepare = prepare
_v126_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v126_base_prepare(weights)
    state['_v126_enabled'] = {}
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    try:
        prelim = {}
        savings = {8: 0.0, 16: 0.0, 32: 0.0}
        for layer in ('qkv_gate', 'o', 'pooler', 'unpooler'):
            for m in (8, 16, 32):
                name = _sk2_name(layer, m)
                if name not in state.get('sk2', {}):
                    continue
                n = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                k = weights[layer].shape[1]
                bitwise, finite, rel_fp64, probe_exact = _v126_validate(
                    weights, state, layer, m)
                info = _v126_resources(layer, m, state)
                gen = torch.Generator(device=weights[layer].device).manual_seed(
                    126700 + m * 19 + sum(map(ord, layer)))
                x = torch.randn((m, k), dtype=torch.bfloat16,
                                device=weights[layer].device, generator=gen)
                data = {'x': x, 'config': {'layer': layer, 'm': m,
                                           'n': n, 'k': k}}
                cand_ms, base_ms = _v102_paired_graph_ms(
                    lambda: _v126_run(x, layer, m, n, k, state),
                    lambda: _v126_base_kernel_fn(data, state),
                    weights[layer].device, repeats=9)
                cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
                core_ratio = ((cand_us - 4.0) / (base_us - 4.0)
                              if base_us > 4.05 else cand_us / base_us)
                ok = (bitwise and finite and rel_fp64 < 5e-3 and probe_exact and
                      info['local_bytes'] == 0 and
                      info['active_ctas_per_sm'] >= 1 and cand_us < base_us and
                      core_ratio <= 0.98)
                prelim[(layer, m)] = ok
                if ok:
                    savings[m] += base_us - cand_us
                print('v126 sk2 slotowner', layer, 'm', m,
                      'resources', info, 'bitwise', bitwise, 'finite', finite,
                      'fp64_rel', rel_fp64, 'probe_exact', probe_exact,
                      'candidate_us', cand_us, 'incumbent_us', base_us,
                      'core_ratio', core_ratio, 'route_ok', ok, flush=True)

        load_us = {8: 37.22, 16: 47.47, 32: 51.69}
        ratios = [max(1e-6, (load_us[m] - savings[m]) / load_us[m])
                  for m in (8, 16, 32)]
        projected_gain = 1.0 - _v126_math.prod(ratios) ** (1.0 / 6.0)
        aggregate_ok = projected_gain >= 0.02
        if aggregate_ok:
            state['_v126_enabled'].update(prelim)
        print('v126 sk2 slotowner projection savings_us', savings,
              'projected_gain', projected_gain,
              'aggregate_2pct_gate', aggregate_ok,
              'enabled', sorted(k for k, v in state['_v126_enabled'].items() if v),
              flush=True)
    except Exception as exc:
        print('v126 sk2 slotowner unavailable', type(exc).__name__,
              str(exc)[:500], flush=True)
        state['_v126_enabled'] = {}
    return state


def kernel_fn(data, state):
    cfg = data['config']
    layer, m, n, k = cfg['layer'], cfg['m'], cfg['n'], cfg['k']
    bucket = 8 if 2 <= m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 0
    if (bucket and layer != 'router' and
            state.get('_v126_enabled', {}).get((layer, bucket), False)):
        return _v126_run(data['x'], layer, m, n, k, state)
    return _v126_base_kernel_fn(data, state)


# v192: broad per-route revalidation against v187.  Every candidate is
# fallback-only unless its own paired cold-L2 adjusted ratio is at most 0.99.
_V134_REPLAY_OVERHEAD_US = 4.0
_V134_REQUIRED_SCORE_GAIN = 0.0
_V134_LOAD_US = {'decode1': 34.65, 'decode16': 47.47, 'decode32': 51.69, 'verify1': 37.22}
_v134_cg_ext = None


def _v134_sk2_l2_src(sm90):
    if not sm90:
        return r'''
void sk2_l2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t,
            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
            int64_t, at::Tensor) { TORCH_CHECK(false, "sk2_l2 needs sm_90"); }
int64_t sk2_l2_info(int64_t, int64_t) { return 0; }
'''
    source = SK2_SRC
    initial_weight = 'bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);'
    refill_weight = 'bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);'
    if source.count(initial_weight) != 1 or source.count(refill_weight) != 1:
        raise RuntimeError('v134 L2 weight-copy marker mismatch')
    source = source.replace(initial_weight, initial_weight.replace(
        'bulk_g2s(', 'v134_bulk_g2s_ef('), 1)
    source = source.replace(refill_weight, refill_weight.replace(
        'bulk_g2s(', 'v134_bulk_g2s_ef('), 1)
    if source.count('tma_load_3d(') != 2:
        raise RuntimeError('v134 L2 activation-copy marker mismatch')
    source = source.replace('tma_load_3d(', 'v134_tma_load_3d_el(')
    source = source.replace('sk2_', 'sk2_l2_').replace(
        'launch_sk2', 'launch_sk2_l2')
    if source.count('void sk2(') != 1:
        raise RuntimeError('v134 L2 wrapper marker mismatch')
    source = source.replace('void sk2(', 'void sk2_l2(', 1)
    helpers = r'''
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void v134_bulk_g2s_ef(
    uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  const uint64_t policy = l2_pol_ef();
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(policy) : "memory");
}
__device__ __forceinline__ void v134_tma_load_3d_el(
    uint32_t dst, const CUtensorMap* map, uint32_t bar,
    int c0, int c1, int c2) {
  const uint64_t policy = l2_pol_el();
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(policy) : "memory");
}
#endif
'''
    source = ('\n#undef SK2_WAIT\n#undef SK2_DELAY_HOOK\n'
              '#undef SK2_DLY_SEED_DECL\n#undef SK2_DLY_SEED_SET\n'
              '#undef SK2_DLY_PICK\n' + helpers + source)
    matches = [(ci, cfg) for ci, cfg in enumerate(SK2_CONFIGS)
               if cfg[1] == 'unpooler' and
               SK2_PICK.get(('unpooler', 32)) == cfg[0]]
    if len(matches) != 1:
        raise RuntimeError('v134 unpooler32 config mismatch')
    ci, cfg = matches[0]
    _, _, _, _, nt, nwr, wsk, upw, nacc, cl = cfg
    hl, nstage = _sk2_geom(cfg, 32)
    mt = 2
    actual_nacc = 1 if (nstage >= 4 and nt == 4) else nacc
    dispatch = (f'if (cfg == {ci} && tokr == 32) {{ '
                f'launch_sk2_l2<{mt}, 32, {nt}, {nwr}, {wsk}, {upw}, '
                f'{nstage}, true, false, {actual_nacc}, {cl}, '
                '__nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, '
                'dg, dbg); return; }')
    source = source.replace('@@SK2_DISPATCH@@', dispatch)
    source += r'''
int64_t sk2_l2_info(int64_t, int64_t) { return 0; }
'''
    return source


CPP_SRC += r'''
void sk2_l2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out,
            torch::Tensor ptab, int64_t pmax, int64_t base4,
            int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr,
            int64_t dg, torch::Tensor dbgt);
int64_t sk2_l2_info(int64_t cfg, int64_t tokr);
'''


def _v134_run_l2(x, state):
    layer, m, n, k = 'unpooler', 32, 8192, 4096
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, _ = state['sk2'][name]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().sk2_l2(stream, x, out, tab, pmax, b4, m, n, k, ci, 32,
                       0, state['dummy'])
    return out


def _v134_cg_source(sm90):
    common = CUDA_SRC.replace('@@MBAR@@', MBAR_HELPERS).replace(
        '@@BODY@@', WG_HELPERS if sm90 else '')
    rg = RGEMV_SRC
    anchor = '__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {'
    helper = r'''
__device__ __forceinline__ uint4 v134_ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 v134_ld2_na(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
'''
    if rg.count(anchor) != 1:
        raise RuntimeError('v134 cgemv helper marker mismatch')
    rg = rg.replace(anchor, helper + anchor, 1)
    replacements = {
        'lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));':
            'lw[r][j] = v134_ld4_na(reinterpret_cast<const uint4*>(L + e0));',
        'cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));':
            'cw[r][j] = v134_ld2_na(reinterpret_cast<const uint2*>(C + e0 / 2));',
    }
    for old, new in replacements.items():
        if rg.count(old) != 1:
            raise RuntimeError('v134 cgemv load marker mismatch')
        rg = rg.replace(old, new, 1)
    return common + rg


def _v134_get_cg_ext():
    global _v134_cg_ext
    if _v134_cg_ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f'{major}{minor}a' if sm90 else f'{major}{minor}'
        cpp = r'''
#include <torch/extension.h>
void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab,
           int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth);
'''
        _v134_cg_ext = load_inline(
            name=f'okc_bf16proj_v134_cgstream_{arch}', cpp_sources=cpp,
            cuda_sources=_v134_cg_source(sm90), functions=['cgemv'],
            extra_cuda_cflags=[
                '-O3', '-lineinfo',
                f'-gencode=arch=compute_{arch},code=sm_{arch}'],
            no_implicit_headers=True, verbose=False)
    return _v134_cg_ext


def _v134_run_pool(x, state):
    cg = state['cg']['pooler']
    out = torch.empty((1, 4096), dtype=torch.bfloat16, device=x.device)
    _v134_get_cg_ext().cgemv(
        cg['L'], cg['C'], x, cg['tab'], cg['pmax'], out,
        cg['base4'], 2, 512)
    return out


_v134_base_prepare = prepare
_v134_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v134_base_prepare(weights)
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    # Measure every composite leg against v105 routing, not v126's aggregate gate.
    state['_v126_enabled'] = {}
    specs = (
        ('qkv_gate', 8, 2304, 8192, 'slot'),
        ('qkv_gate', 16, 2304, 8192, 'slot'),
        ('o', 16, 8192, 1024, 'slot'),
        ('o', 32, 8192, 1024, 'slot'),
        ('pooler', 32, 4096, 8192, 'slot'),
        ('unpooler', 16, 8192, 4096, 'slot'),
        ('unpooler', 32, 8192, 4096, 'l2'),
        ('pooler', 1, 4096, 8192, 'pool'),
    )
    trials = {}
    savings = {'decode1': 0.0, 'decode16': 0.0, 'decode32': 0.0, 'verify1': 0.0}
    try:
        for index, (layer, m, n, k, route) in enumerate(specs):
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(134000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': m, 'n': n, 'k': k}}
            if route == 'slot':
                run = lambda: _v126_run(x, layer, m, n, k, state)
            elif route == 'l2':
                run = lambda: _v134_run_l2(x, state)
            else:
                run = lambda: _v134_run_pool(x, state)
            candidate = run()
            incumbent = _v134_base_kernel_fn(data, state)
            torch.cuda.synchronize(w.device)
            bitwise = bool(torch.equal(candidate, incumbent))
            finite = bool(torch.isfinite(candidate).all())
            cand_ms, base_ms = _v102_paired_graph_ms(
                run, lambda: _v134_base_kernel_fn(data, state),
                w.device, repeats=13)
            cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
            cand_core = max(cand_us - _V134_REPLAY_OVERHEAD_US, 1e-6)
            base_core = max(base_us - _V134_REPLAY_OVERHEAD_US, 1e-6)
            core_ratio = cand_core / base_core
            required = 0.99
            eligible = (bitwise and finite and cand_us < base_us and
                        core_ratio <= required)
            trials[(layer, m)] = {
                'route': route, 'eligible': eligible,
                'candidate_us': cand_us, 'incumbent_us': base_us,
                'core_ratio': core_ratio,
            }
            load = ('decode1' if m == 1 else 'verify1' if m == 8 else
                    'decode16' if m == 16 else 'decode32')
            if eligible:
                savings[load] += base_us - cand_us
            print('v134 composite leg', layer, m, route,
                  'bitwise', bitwise, 'finite', finite,
                  'candidate_us', cand_us, 'incumbent_us', base_us,
                  'core_ratio', core_ratio, 'eligible', eligible, flush=True)
        remaining = 1.0
        for load, old_us in _V134_LOAD_US.items():
            remaining *= max(old_us - savings[load], 1e-6) / old_us
        projected_gain = 1.0 - remaining ** (1.0 / 6.0)
        global_enable = (len(trials) == len(specs) and
                         projected_gain >= _V134_REQUIRED_SCORE_GAIN)
        state['_v134_composite'] = {
            key: trial['route'] for key, trial in trials.items()
            if global_enable and trial['eligible']
        }
        print('v134 composite savings_us', savings,
              'projected_score_gain', projected_gain,
              'required', _V134_REQUIRED_SCORE_GAIN,
              'global_enable', global_enable,
              'enabled', sorted(state['_v134_composite']), flush=True)
    except Exception as exc:
        state['_v134_composite'] = {}
        print('v134 composite unavailable', type(exc).__name__, str(exc),
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    key = (cfg['layer'], cfg['m'])
    route = state.get('_v134_composite', {}).get(key)
    if route == 'slot':
        return _v126_run(data['x'], cfg['layer'], cfg['m'], cfg['n'],
                         cfg['k'], state)
    if route == 'l2':
        return _v134_run_l2(data['x'], state)
    if route == 'pool':
        return _v134_run_pool(data['x'], state)
    return _v134_base_kernel_fn(data, state)
# ---------------------------------------------------- v197: proven QKV/m32 route.
_v197_base_prepare = prepare
_v197_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v197_base_prepare(weights)
    state["v197_qkv32_plan"] = None
    vstate = state.get("v150")
    source = None if vstate is None else vstate.get("plans", {}).get("qkv_gate")
    if source is None:
        return state
    try:
        plan = dict(source)
        plan["variant"] = 2
        if _v158_validate(vstate, plan, weights["qkv_gate"], "qkv_gate", 32):
            state["v197_qkv32_plan"] = plan
            print("v197 qkv32 enabled stage5 policy9437219", flush=True)
        else:
            print("v197 qkv32 validator rejected", flush=True)
    except Exception as exc:
        state["v197_qkv32_plan"] = None
        print("v197 qkv32 unavailable", type(exc).__name__, str(exc)[:240],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    plan = state.get("v197_qkv32_plan")
    if cfg["m"] == 32 and cfg["layer"] == "qkv_gate" and plan is not None:
        return _v158_call(state["v150"], plan, data["x"])
    return _v197_base_kernel_fn(data, state)
# -------------------------------------------- v198: repeated exact micro-wins.
_V198_PINNED = (
    ('qkv_gate', 8, 2304, 8192, 'slot'),
    ('qkv_gate', 16, 2304, 8192, 'slot'),
    ('o', 16, 8192, 1024, 'slot'),
    ('unpooler', 32, 8192, 4096, 'l2'),
    ('pooler', 1, 4096, 8192, 'pool'),
)
_v198_base_prepare = prepare


def prepare(weights):
    state = _v198_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    for index, (layer, m, n, k, route) in enumerate(_V198_PINNED):
        try:
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(198000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': m, 'n': n, 'k': k}}
            if route == 'slot':
                candidate = _v126_run(x, layer, m, n, k, state)
            elif route == 'l2':
                candidate = _v134_run_l2(x, state)
            else:
                candidate = _v134_run_pool(x, state)
            incumbent = _v134_base_kernel_fn(data, state)
            torch.cuda.synchronize(w.device)
            exact = bool(torch.equal(candidate, incumbent))
            finite = bool(torch.isfinite(candidate).all())
            if exact and finite:
                routes[(layer, m)] = route
            else:
                routes.pop((layer, m), None)
            print('v198 pin', layer, m, route, 'bitwise', exact,
                  'finite', finite, 'enabled', exact and finite, flush=True)
        except Exception as exc:
            routes.pop((layer, m), None)
            print('v198 pin unavailable', layer, m, type(exc).__name__,
                  str(exc)[:240], flush=True)
    state['_v134_composite'] = routes
    print('v198 pinned composite enabled', sorted(routes), flush=True)
    return state
# --------------------------- v205: repeated exact O/m32 micro-route win.
# The same readable slot-owner kernel was bitwise exact in every recent run and
# won the last two official paired comparisons.  Revalidate it once more on
# this H200, then keep it independent of the noisy sub-microsecond timing gate.
_v205_base_prepare = prepare


def prepare(weights):
    state = _v205_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    w = weights.get('o')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        m, n, k = 32, 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(205032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': m, 'n': n, 'k': k}}
        candidate = _v126_run(x, 'o', m, n, k, state)
        incumbent = _v134_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        exact = bool(torch.equal(candidate, incumbent))
        finite = bool(torch.isfinite(candidate).all())
        probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
        cols = (0, k // 3, (2 * k) // 3, k - 1)
        for row, col in enumerate(cols):
            probe[row, col] = 1
        got = _v126_run(probe, 'o', m, n, k, state)
        probe_exact = all(torch.equal(got[row], w[:, col])
                          for row, col in enumerate(cols))
        if exact and finite and probe_exact:
            routes[('o', 32)] = 'slot'
        else:
            routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v205 pin o 32 slot bitwise', exact, 'finite', finite,
              'probe_exact', probe_exact,
              'enabled', exact and finite and probe_exact, flush=True)
    except Exception as exc:
        routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v205 pin unavailable', type(exc).__name__, str(exc)[:300],
              flush=True)
    return state
# v207 independently repeats the O/m32 exactness guard before scored dispatch.
_v207_base_prepare = prepare


def prepare(weights):
    state = _v207_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    if routes.get(('o', 32)) != 'slot':
        return state
    w = weights.get('o')
    try:
        m, n, k = 32, 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(207032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': m, 'n': n, 'k': k}}
        candidate = _v126_run(x, 'o', m, n, k, state)
        incumbent = _v134_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        exact = bool(torch.equal(candidate, incumbent))
        finite = bool(torch.isfinite(candidate).all())
        if not (exact and finite):
            routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v207 second guard o 32 bitwise', exact, 'finite', finite,
              'enabled', exact and finite, flush=True)
    except Exception as exc:
        routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v207 guard unavailable', type(exc).__name__, str(exc)[:240],
              flush=True)
    return state
# v209 adds a third independent validation activation for the pinned O/m32 path.
_v209_base_prepare = prepare


def prepare(weights):
    state = _v209_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    if routes.get(('o', 32)) != 'slot':
        return state
    w = weights.get('o')
    try:
        m, n, k = 32, 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(209032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': m, 'n': n, 'k': k}}
        candidate = _v126_run(x, 'o', m, n, k, state)
        incumbent = _v134_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        exact = bool(torch.equal(candidate, incumbent))
        finite = bool(torch.isfinite(candidate).all())
        if not (exact and finite):
            routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v209 third guard o 32 bitwise', exact, 'finite', finite,
              'enabled', exact and finite, flush=True)
    except Exception as exc:
        routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v209 guard unavailable', type(exc).__name__, str(exc)[:240],
              flush=True)
    return state
# v210: v208's official H200 tournament selected the ordinary pooler/m32
# slot-owner at a 0.9816 replay-adjusted ratio.  Revalidate numerical quality
# and exact coefficient recovery before making that existing kernel stable.
_v210_base_prepare = prepare


def prepare(weights):
    state = _v210_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    w = weights.get('pooler')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        m, n, k = 32, 4096, 8192
        gen = torch.Generator(device=w.device).manual_seed(210032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        out = _v126_run(x, 'pooler', m, n, k, state)
        torch.cuda.synchronize(w.device)
        finite = bool(torch.isfinite(out).all())
        rows = torch.linspace(0, n - 1, steps=64, device=w.device).long()
        ref = x.double() @ w[rows].double().t()
        diff = out[:, rows].double() - ref
        rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
        den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        worst = float((diff.norm(dim=1) / den).max())
        probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
        cols = (0, k // 5, k // 3, k // 2, k - 1)
        for row, col in enumerate(cols):
            probe[row, col] = 1
        got = _v126_run(probe, 'pooler', m, n, k, state)
        probe_exact = all(torch.equal(got[row], w[:, col])
                          for row, col in enumerate(cols))
        valid = finite and probe_exact and rel < 0.005 and worst < 0.02
        if valid:
            routes[('pooler', 32)] = 'slot'
        else:
            routes.pop(('pooler', 32), None)
        state['_v134_composite'] = routes
        print('v210 pin pooler 32 finite', finite,
              'probe_exact', probe_exact, 'fp64', rel, worst,
              'enabled', valid, flush=True)
    except Exception as exc:
        routes.pop(('pooler', 32), None)
        state['_v134_composite'] = routes
        print('v210 pin unavailable', type(exc).__name__, str(exc)[:300],
              flush=True)
    return state
# v221: independently validate all seven promoted micro-routes on fresh
# activations.  The already validated v197 QKV32 plan retains its own exact
# prepare-time validator.  Passing this guard preserves v210 timed dispatch.
_V221_ROUTES = (
    ('qkv_gate', 8, 2304, 8192, 'slot'),
    ('qkv_gate', 16, 2304, 8192, 'slot'),
    ('o', 16, 8192, 1024, 'slot'),
    ('o', 32, 8192, 1024, 'slot'),
    ('pooler', 1, 4096, 8192, 'pool'),
    ('pooler', 32, 4096, 8192, 'slot'),
    ('unpooler', 32, 8192, 4096, 'l2'),
)
_v221_base_prepare = prepare


def _v221_run(route, x, layer, m, n, k, state):
    if route == 'slot':
        return _v126_run(x, layer, m, n, k, state)
    if route == 'l2':
        return _v134_run_l2(x, state)
    return _v134_run_pool(x, state)


def prepare(weights):
    state = _v221_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    for index, (layer, m, n, k, route) in enumerate(_V221_ROUTES):
        key = (layer, m)
        if routes.get(key) != route:
            print('v221 full guard route absent', key, route, flush=True)
            continue
        try:
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(221000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            out = _v221_run(route, x, layer, m, n, k, state)
            torch.cuda.synchronize(w.device)
            finite = bool(torch.isfinite(out).all())
            rows = torch.arange(13 + index, n, 97, device=w.device)[:32]
            ref = x.double() @ w[rows].double().t()
            diff = out[:, rows].double() - ref
            rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
            den = torch.maximum(ref.norm(dim=1),
                                0.1 * ref.norm(dim=1).mean())
            worst = float((diff.norm(dim=1) / den).max())
            cols = (0, k // 7, k // 3, (2 * k) // 3, k - 1)
            probe_exact = True
            if m == 1:
                for col in cols:
                    probe = torch.zeros((1, k), dtype=torch.bfloat16,
                                        device=w.device)
                    probe[0, col] = 1
                    got = _v221_run(route, probe, layer, m, n, k, state)
                    probe_exact = probe_exact and bool(
                        torch.equal(got[0], w[:n, col]))
            else:
                probe = torch.zeros((m, k), dtype=torch.bfloat16,
                                    device=w.device)
                for row, col in enumerate(cols):
                    probe[row, col] = 1
                got = _v221_run(route, probe, layer, m, n, k, state)
                probe_exact = all(torch.equal(got[row], w[:n, col])
                                  for row, col in enumerate(cols))
            valid = finite and probe_exact and rel < 0.005 and worst < 0.02
            if not valid:
                routes.pop(key, None)
            print('v221 full guard', layer, m, route, 'finite', finite,
                  'probe_exact', probe_exact, 'fp64', rel, worst,
                  'enabled', valid, flush=True)
        except Exception as exc:
            routes.pop(key, None)
            print('v221 full guard unavailable', layer, m,
                  type(exc).__name__, str(exc)[:300], flush=True)
    state['_v134_composite'] = routes
    print('v221 full guard enabled', sorted(routes), flush=True)
    return state
