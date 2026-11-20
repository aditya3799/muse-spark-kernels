"""Cold-L2 CUDA-graph sweep for MODE-2 boundary cache policies.

This is an isolated local probe.  It reproduces the scored CUDA boundary's
floating-point operation order while varying only cache operators for the two
BF16 inputs and FP32/BF16 outputs.  It never modifies submission.py.
"""

from __future__ import annotations

import os
import statistics

import torch
from torch.utils.cpp_extension import load_inline


CPP = r"""
#include <torch/extension.h>
void run_boundary(torch::Tensor branch, torch::Tensor br1, torch::Tensor rio,
                  torch::Tensor inv_gpre1, torch::Tensor gpost,
                  torch::Tensor alpha, torch::Tensor beta, torch::Tensor gpre,
                  torch::Tensor sout, torch::Tensor nout, int64_t mode);
"""


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int H = 8192;
constexpr int TPB = 512;
constexpr int C = H / (TPB * 8);
constexpr int NW = TPB / 32;

template <bool STREAM>
__device__ __forceinline__ uint4 load_u4(const uint4* p) {
  if constexpr (STREAM) return __ldcs(p);
  return *p;
}

template <bool STREAM>
__device__ __forceinline__ void store_f4(float4* p, float4 v) {
  if constexpr (STREAM) __stcs(p, v);
  else *p = v;
}

template <bool STREAM>
__device__ __forceinline__ void store_u4(uint4* p, uint4 v) {
  if constexpr (STREAM) __stcs(p, v);
  else *p = v;
}

__device__ __forceinline__ float4 gain4(const __nv_bfloat16* p) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 a = __bfloat1622float2(h[0]);
  const float2 b = __bfloat1622float2(h[1]);
  return make_float4(1.f + a.x, 1.f + a.y, 1.f + b.x, 1.f + b.y);
}

template <bool BRANCH_CS, bool BR1_CS, bool SOUT_CS, bool NOUT_CS>
__global__ __launch_bounds__(TPB, 2)
void boundary_mode2(const __nv_bfloat16* __restrict__ branch,
                    const __nv_bfloat16* __restrict__ br1,
                    const float* __restrict__ rio,
                    const float* __restrict__ inv_gpre1,
                    const __nv_bfloat16* __restrict__ gpost,
                    const float* __restrict__ alpha,
                    const float* __restrict__ beta,
                    const __nv_bfloat16* __restrict__ gpre,
                    float* __restrict__ sout,
                    __nv_bfloat16* __restrict__ nout) {
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float red[2][NW];
  float b[C * 8], s[C * 8];
  const __nv_bfloat16* bp = branch + row * H;
  const __nv_bfloat16* hp = br1 + row * H;

#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const uint4 vb = load_u4<BRANCH_CS>(reinterpret_cast<const uint4*>(bp + off));
    const uint4 vh = load_u4<BR1_CS>(reinterpret_cast<const uint4*>(hp + off));
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&vb);
    const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&vh);
    const float row_rms = rio[row];
    const float4 i0 = __ldg(reinterpret_cast<const float4*>(inv_gpre1 + off));
    const float4 i1 = __ldg(reinterpret_cast<const float4*>(inv_gpre1 + off + 4));
    const float iv[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 fb = __bfloat1622float2(hb[i]);
      const float2 fh = __bfloat1622float2(hh[i]);
      b[c * 8 + 2 * i] = fb.x;
      b[c * 8 + 2 * i + 1] = fb.y;
      s[c * 8 + 2 * i] = fh.x * (row_rms * iv[2 * i]);
      s[c * 8 + 2 * i + 1] = fh.y * (row_rms * iv[2 * i + 1]);
    }
  }

  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 g0 = gain4(gpost + off);
    const float4 g1 = gain4(gpost + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      b[c * 8 + i] *= gv[i];
      ss += b[c * 8 + i] * b[c * 8 + i];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[0][tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += red[0][i];
  const float r1 = rsqrtf(tot / H + 1e-8f);

  float ss2 = 0.f;
  float* so = sout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha + off));
    const float4 a1 = __ldg(reinterpret_cast<const float4*>(alpha + off + 4));
    const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta + off));
    const float4 e1 = __ldg(reinterpret_cast<const float4*>(beta + off + 4));
    const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float ev[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s[c * 8 + i] = __fmaf_rn(av[i], s[c * 8 + i], ev[i] * (b[c * 8 + i] * r1));
      ss2 += s[c * 8 + i] * s[c * 8 + i];
    }
    const float4 o0 = make_float4(s[c * 8 + 0], s[c * 8 + 1], s[c * 8 + 2], s[c * 8 + 3]);
    const float4 o1 = make_float4(s[c * 8 + 4], s[c * 8 + 5], s[c * 8 + 6], s[c * 8 + 7]);
    store_f4<SOUT_CS>(reinterpret_cast<float4*>(so + off), o0);
    store_f4<SOUT_CS>(reinterpret_cast<float4*>(so + off + 4), o1);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss2 += __shfl_xor_sync(0xffffffffu, ss2, o);
  if ((tid & 31) == 0) red[1][tid >> 5] = ss2;
  __syncthreads();
  float tot2 = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot2 += red[1][i];
  const float r2 = rsqrtf(tot2 / H + 1e-5f);

  __nv_bfloat16* no = nout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 g0 = gain4(gpre + off);
    const float4 g1 = gain4(gpre + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    uint4 out;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ho[i] = __floats2bfloat162_rn(s[c * 8 + 2 * i] * r2 * gv[2 * i],
                                    s[c * 8 + 2 * i + 1] * r2 * gv[2 * i + 1]);
    }
    store_u4<NOUT_CS>(reinterpret_cast<uint4*>(no + off), out);
  }
}

template <bool B, bool H1, bool SO, bool NO>
void launch(const __nv_bfloat16* branch, const __nv_bfloat16* br1, const float* rio,
            const float* inv_gpre1, const __nv_bfloat16* gpost, const float* alpha,
            const float* beta, const __nv_bfloat16* gpre, float* sout,
            __nv_bfloat16* nout, int T, cudaStream_t stream) {
  boundary_mode2<B, H1, SO, NO><<<T, TPB, 0, stream>>>(
      branch, br1, rio, inv_gpre1, gpost, alpha, beta, gpre, sout, nout);
}

}  // namespace

void run_boundary(torch::Tensor branch, torch::Tensor br1, torch::Tensor rio,
                  torch::Tensor inv_gpre1, torch::Tensor gpost,
                  torch::Tensor alpha, torch::Tensor beta, torch::Tensor gpre,
                  torch::Tensor sout, torch::Tensor nout, int64_t mode) {
  const int T = branch.size(0);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* b = reinterpret_cast<const __nv_bfloat16*>(branch.data_ptr());
  const auto* h = reinterpret_cast<const __nv_bfloat16*>(br1.data_ptr());
  const auto* gp = reinterpret_cast<const __nv_bfloat16*>(gpost.data_ptr());
  const auto* gn = reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr());
  auto* no = reinterpret_cast<__nv_bfloat16*>(nout.data_ptr());
  switch (mode) {
    case 1: launch<true, false, false, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 2: launch<false, true, false, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 4: launch<false, false, true, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 5: launch<true, false, true, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 6: launch<false, true, true, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 7: launch<true, true, true, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    case 15: launch<true, true, true, true>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
    default: launch<false, false, false, false>(b,h,rio.data_ptr<float>(),inv_gpre1.data_ptr<float>(),gp,alpha.data_ptr<float>(),beta.data_ptr<float>(),gn,sout.data_ptr<float>(),no,T,stream); break;
  }
}
"""


MODES = {
    0: "normal",
    1: "branch.cs",
    2: "br1.cs",
    4: "sout.cs",
    5: "branch+sout.cs",
    6: "br1+sout.cs",
    7: "both-in+sout.cs",
    15: "all.cs",
}


def make_bf16(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    rows = shape[0]
    for lo in range(0, rows, 256):
        n = min(256, rows - lo)
        out[lo : lo + n].copy_((torch.randn((n, shape[1]), device="cuda") * scale).to(torch.bfloat16))
    return out


def launch(mod, args, mode: int) -> None:
    mod.run_boundary(*args, mode)


def graph_times(mod, args, mode: int, flush: torch.Tensor, reps: int = 17) -> list[float]:
    for _ in range(3):
        launch(mod, args, mode)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(mod, args, mode)
    samples = []
    for _ in range(reps):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(os.path.dirname(__file__), ".torch_extensions"))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; SM {props.major}.{props.minor}; {props.total_memory / 2**30:.1f} GiB")
    mod = load_inline(
        "probe_boundary_cache_policy_v1",
        cpp_sources=CPP,
        cuda_sources=CUDA,
        functions=["run_boundary"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )

    # T=3012 catches small-shape behavior; T=16384 matches the dominant scored path.
    for T in (3012, 16384):
        print(f"\nT={T}")
        branch = make_bf16((T, H := 8192), 0.20)
        br1 = make_bf16((T, H), 0.25)
        rio = torch.rand((T,), device="cuda", dtype=torch.float32) * 0.4 + 0.8
        inv_gpre1 = torch.rand((H,), device="cuda", dtype=torch.float32) * 0.2 + 0.9
        gpost = (torch.randn((H,), device="cuda") * 0.03).to(torch.bfloat16)
        alpha = torch.rand((H,), device="cuda", dtype=torch.float32) * 0.2 + 0.9
        beta = torch.rand((H,), device="cuda", dtype=torch.float32) * 0.2 + 0.9
        gpre = (torch.randn((H,), device="cuda") * 0.03).to(torch.bfloat16)
        sout = torch.empty((T, H), device="cuda", dtype=torch.float32)
        nout = torch.empty((T, H), device="cuda", dtype=torch.bfloat16)
        args = (branch, br1, rio, inv_gpre1, gpost, alpha, beta, gpre, sout, nout)

        launch(mod, args, 0)
        torch.cuda.synchronize()
        ref_s = sout.clone()
        ref_n = nout.clone()
        for mode, name in MODES.items():
            launch(mod, args, mode)
            torch.cuda.synchronize()
            ds = float((sout - ref_s).abs().max())
            dn = float((nout.float() - ref_n.float()).abs().max())
            if ds != 0.0 or dn != 0.0:
                raise AssertionError(f"{name}: output mismatch max stream={ds} hidden={dn}")

        flush = torch.empty((128 << 20,), device="cuda", dtype=torch.uint8)
        rows = []
        for mode, name in MODES.items():
            samples = graph_times(mod, args, mode, flush)
            rows.append((statistics.median(samples), min(samples), max(samples), name))
        base = next(x[0] for x in rows if x[3] == "normal")
        for med, best, worst, name in sorted(rows):
            print(f"{name:20s} median {med:9.1f} us  best {best:9.1f}  worst {worst:9.1f}  delta {(med/base-1)*100:+6.2f}%")


if __name__ == "__main__":
    main()
