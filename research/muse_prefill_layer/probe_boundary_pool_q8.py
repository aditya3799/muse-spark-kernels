"""Parity/timing probe for the fused first-boundary + pooler row-quantization path."""

import hashlib

import torch
from torch.utils.cpp_extension import load_inline

import submission as sub


def build_local():
    """Build only the exact boundary specialization; the full source needs a C++20 host compiler."""
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}" + ("a" if cap[0] == 9 else "")
    full = sub._CUDA_SRC
    q0 = full.index("DEVI uint32_t qk_e4m3_sw")
    q1 = full.index("DEVI void qk_load8", q0)
    b0 = full.index("// Residual boundary")
    b1 = full.index("}  // namespace okc_moe", b0)
    cuda = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define DEVI __device__ __forceinline__
__device__ __forceinline__ void pdl_enter() {}
namespace okc_moe {
DEVI float4 g1p4(const __nv_bfloat16* g) {
  const uint2 v = __ldg(reinterpret_cast<const uint2*>(g));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  return make_float4(1.f + a.x, 1.f + a.y, 1.f + b.x, 1.f + b.y);
}
''' + full[q0:q1] + full[b0:b1] + r'''
}  // namespace okc_moe
using namespace okc_moe;
void k_boundary_ref(const void* branch, const float* stream, const float* alpha, const float* beta, const void* gpre,
                    float* sout, void* nout, int T, int mode, float* rio, cudaStream_t st) {
  if (mode == 1)
    boundary_kernel<512, false, false, 1><<<T, 512, 0, st>>>((const __nv_bfloat16*)branch, stream, nullptr,
        alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, nullptr, 0, nullptr, rio, nullptr, nullptr);
  else
    boundary_kernel<512, false, false, 0><<<T, 512, 0, st>>>((const __nv_bfloat16*)branch, stream, nullptr,
        alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout);
}
void k_boundary_q8(const void* branch, const float* stream, const float* alpha, const float* beta, const void* gpre,
                   float* sout, void* nout, int T, int mode, float* rio, void* qout, float* qscale, cudaStream_t st) {
  if (mode == 1)
    boundary_kernel<512, false, false, 1, true><<<T, 512, 0, st>>>((const __nv_bfloat16*)branch, stream, nullptr,
        alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, nullptr, 0, nullptr, rio, nullptr, nullptr,
        (uint8_t*)qout, qscale);
  else
    boundary_kernel<512, false, false, 0, true><<<T, 512, 0, st>>>((const __nv_bfloat16*)branch, stream, nullptr,
        alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, nullptr, 0, nullptr, nullptr, nullptr, nullptr,
        (uint8_t*)qout, qscale);
}
'''
    cpp = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_boundary_ref(const void*, const float*, const float*, const float*, const void*, float*, void*, int, int, float*, cudaStream_t);
void k_boundary_q8(const void*, const float*, const float*, const float*, const void*, float*, void*, int, int, float*, void*, float*, cudaStream_t);
void boundary_ref(torch::Tensor branch, torch::Tensor stream, torch::Tensor alpha, torch::Tensor beta, torch::Tensor gpre,
                  torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor rio) {
  k_boundary_ref(branch.data_ptr(), stream.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), gpre.data_ptr(),
                 sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, rio.data_ptr<float>(),
                 c10::cuda::getCurrentCUDAStream());
}
void boundary_q8(torch::Tensor branch, torch::Tensor stream, torch::Tensor alpha, torch::Tensor beta, torch::Tensor gpre,
                 torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor rio, torch::Tensor qout, torch::Tensor qscale) {
  k_boundary_q8(branch.data_ptr(), stream.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), gpre.data_ptr(),
                sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, rio.data_ptr<float>(), qout.data_ptr(),
                qscale.data_ptr<float>(), c10::cuda::getCurrentCUDAStream());
}
'''
    name = "okc_boundary_q8_local_" + hashlib.sha1((cuda + cpp + arch).encode()).hexdigest()[:16]
    return load_inline(name, cpp_sources=cpp, cuda_sources=cuda, functions=["boundary_ref", "boundary_q8"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                          "-std=c++17", "--use_fast_math"])


def rowq8(x):
    out = torch.empty_like(x, dtype=sub.FP8)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    sub._rowq8_kernel[(x.shape[0],)](x, out, scale, H=sub.HIDDEN, num_warps=16)
    return out, scale


def compare(name, lhs, rhs):
    if lhs.dtype == sub.FP8:
        same = torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
        diff = int((lhs.view(torch.uint8) != rhs.view(torch.uint8)).sum())
        print(f"{name}: exact={same} differing_bytes={diff}")
        return same
    same = torch.equal(lhs, rhs)
    max_diff = float((lhs.float() - rhs.float()).abs().max())
    print(f"{name}: exact={same} max_abs={max_diff:.9g}")
    return same


def bench(fn, reps=100):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    beg.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    end.synchronize()
    return beg.elapsed_time(end) * 1000.0 / reps


def main():
    torch.manual_seed(20261030)
    dev = "cuda"
    mod = build_local()
    ok = True
    for mode, tokens in ((0, 16), (1, 16)):
        branch = torch.randn((tokens, sub.HIDDEN), device=dev, dtype=torch.bfloat16)
        stream = torch.randn((tokens, sub.HIDDEN), device=dev, dtype=torch.float32)
        alpha = torch.randn((sub.HIDDEN,), device=dev, dtype=torch.float32) * 0.1 + 0.9
        beta = torch.randn((sub.HIDDEN,), device=dev, dtype=torch.float32) * 0.1 + 0.8
        gpre = torch.randn((sub.HIDDEN,), device=dev, dtype=torch.bfloat16) * 0.05
        sout_ref = torch.empty_like(stream)
        h_ref = torch.empty_like(branch)
        rio_ref = torch.empty((tokens,), device=dev, dtype=torch.float32)
        mod.boundary_ref(branch, stream, alpha, beta, gpre, sout_ref, h_ref, mode, rio_ref)
        q_ref, sc_ref = rowq8(h_ref)

        sout_new = torch.empty_like(stream)
        h_new = torch.empty_like(branch)
        rio_new = torch.empty((tokens,), device=dev, dtype=torch.float32)
        q_new = torch.empty_like(branch, dtype=sub.FP8)
        sc_new = torch.empty((tokens, 1), device=dev, dtype=torch.float32)
        mod.boundary_q8(branch, stream, alpha, beta, gpre, sout_new, h_new, mode, rio_new, q_new, sc_new)
        torch.cuda.synchronize()

        print(f"mode={mode}")
        ok &= compare("bf16 h", h_ref, h_new)
        ok &= compare("fp8 h", q_ref, q_new)
        ok &= compare("row scale", sc_ref, sc_new)
        ok &= compare("stream" if mode == 0 else "rms factor", sout_ref if mode == 0 else rio_ref,
                      sout_new if mode == 0 else rio_new)

    # A modest local timing shape: enough rows to expose the removed launch/read without exhausting a laptop GPU.
    tokens = 256
    branch = torch.randn((tokens, sub.HIDDEN), device=dev, dtype=torch.bfloat16)
    stream = torch.randn((tokens, sub.HIDDEN), device=dev, dtype=torch.float32)
    alpha = torch.ones((sub.HIDDEN,), device=dev, dtype=torch.float32)
    beta = torch.ones((sub.HIDDEN,), device=dev, dtype=torch.float32)
    gpre = torch.zeros((sub.HIDDEN,), device=dev, dtype=torch.bfloat16)
    sout = torch.empty_like(stream)
    h = torch.empty_like(branch)
    q = torch.empty_like(branch, dtype=sub.FP8)
    sc = torch.empty((tokens, 1), device=dev, dtype=torch.float32)
    rio = torch.empty((tokens,), device=dev, dtype=torch.float32)

    def old_path():
        mod.boundary_ref(branch, stream, alpha, beta, gpre, sout, h, 0, rio)
        sub._rowq8_kernel[(tokens,)](h, q, sc, H=sub.HIDDEN, num_warps=16)

    old_us = bench(old_path)
    new_us = bench(lambda: mod.boundary_q8(branch, stream, alpha, beta, gpre, sout, h, 0, rio, q, sc))
    print(f"local T={tokens}: old={old_us:.3f}us fused={new_us:.3f}us ratio={new_us / old_us:.4f}")
    if not ok:
        raise SystemExit("parity failed")


if __name__ == "__main__":
    main()
