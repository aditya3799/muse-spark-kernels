# okc: problem=muse_bf16_proj
"""Starter: the projections in plain PyTorch, correct but slow.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch
import torch.nn.functional as F
from flashinfer.gemm import tinygemm_bf16
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
from torch.utils.cpp_extension import load_inline


_ROUTER_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>
#include <unordered_map>

namespace {

constexpr int kRouterN = 256;
constexpr int kThreads = 256;
constexpr size_t kLtWorkspaceBytes = 32ull << 20;

struct LtChoice {
  cublasLtMatmulAlgo_t algo;
  size_t workspace_bytes;
};

std::unordered_map<unsigned long long, LtChoice> g_lt_choices;
cublasLtHandle_t g_lt_handle = nullptr;

unsigned long long lt_key(int m, int n, int k) {
  return (static_cast<unsigned long long>(m) << 32) |
         (static_cast<unsigned long long>(n) << 16) |
         static_cast<unsigned long long>(k);
}

void check_lt(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, " failed with cuBLASLt status ", static_cast<int>(status));
}

void ensure_lt_handle() {
  if (g_lt_handle == nullptr) check_lt(cublasLtCreate(&g_lt_handle), "cublasLtCreate");
}

struct LtDescriptors {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;

  LtDescriptors(int m, int n, int k) {
    check_lt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    cublasOperation_t no_transpose = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose, sizeof(no_transpose)), "set TRANSA");
    check_lt(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose)), "set TRANSB");
    // Row-major out = x[M,K] @ weight_t[K,N] is column-major
    // out^T[N,M] = weight_t^T[N,K] @ x^T[K,M].
    check_lt(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, n, k, n), "create A layout");
    check_lt(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k), "create B layout");
    check_lt(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n), "create C layout");
  }

  ~LtDescriptors() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

cublasStatus_t run_lt(
    const torch::Tensor& x,
    const torch::Tensor& weight_t,
    torch::Tensor& output,
    const torch::Tensor& workspace,
    const cublasLtMatmulAlgo_t* algo,
    size_t workspace_bytes,
    LtDescriptors& desc) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return cublasLtMatmul(
      g_lt_handle, desc.op, &alpha,
      weight_t.data_ptr<at::BFloat16>(), desc.a,
      x.data_ptr<at::BFloat16>(), desc.b,
      &beta, output.data_ptr<at::BFloat16>(), desc.c,
      output.data_ptr<at::BFloat16>(), desc.c,
      algo, workspace.data_ptr(), workspace_bytes, stream);
}

template <int M>
__global__ __launch_bounds__(kThreads)
void router_fp32_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int k) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float acc[M];
#pragma unroll
  for (int i = 0; i < M; ++i) acc[i] = 0.0f;

  const float* w = weight + static_cast<long long>(row) * k;
  for (int col = threadIdx.x; col < k; col += kThreads) {
    const float value = __ldg(w + col);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      acc[i] = fmaf(__bfloat162float(x[static_cast<long long>(i) * k + col]), value, acc[i]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      acc[i] += __shfl_down_sync(0xffffffff, acc[i], offset);
    }
  }

  __shared__ float partial[M][8];
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < M; ++i) partial[i][warp] = acc[i];
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      float value = lane < 8 ? partial[i][lane] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
      }
      if (lane == 0) output[static_cast<long long>(i) * kRouterN + row] = value;
    }
  }
}

template <int M>
void launch_router(const torch::Tensor& x, const torch::Tensor& weight, torch::Tensor& output) {
  const int k = static_cast<int>(x.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  router_fp32_gemv_kernel<M><<<kRouterN, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      weight.data_ptr<float>(), output.data_ptr<float>(), k);
}

}  // namespace

torch::Tensor router_fp32_gemv(torch::Tensor x, torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "router tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "router input must be BF16");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "router weight must be FP32");
  TORCH_CHECK(weight.size(0) == kRouterN && x.size(1) == weight.size(1), "unexpected router shape");
  auto output = torch::empty({x.size(0), kRouterN}, x.options().dtype(torch::kFloat32));
  switch (x.size(0)) {
    case 1: launch_router<1>(x, weight, output); break;
    case 2: launch_router<2>(x, weight, output); break;
    case 3: launch_router<3>(x, weight, output); break;
    case 4: launch_router<4>(x, weight, output); break;
    case 5: launch_router<5>(x, weight, output); break;
    case 6: launch_router<6>(x, weight, output); break;
    case 7: launch_router<7>(x, weight, output); break;
    case 8: launch_router<8>(x, weight, output); break;
    default: TORCH_CHECK(false, "router_fp32_gemv supports 1..8 rows");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void lt_tune(torch::Tensor x, torch::Tensor weight_t, torch::Tensor workspace) {
  ensure_lt_handle();
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(weight_t.size(1));
  TORCH_CHECK(x.is_cuda() && weight_t.is_cuda() && workspace.is_cuda(), "Lt tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && weight_t.scalar_type() == torch::kBFloat16,
              "Lt operands must be BF16");
  TORCH_CHECK(x.is_contiguous() && weight_t.is_contiguous(), "Lt operands must be contiguous");
  TORCH_CHECK(weight_t.size(0) == k && workspace.numel() >= static_cast<long long>(kLtWorkspaceBytes),
              "unexpected Lt shapes or workspace");

  LtDescriptors desc(m, n, k);
  cublasLtMatmulPreference_t preference = nullptr;
  check_lt(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
  size_t workspace_limit = kLtWorkspaceBytes;
  check_lt(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_limit, sizeof(workspace_limit)), "set workspace preference");

  constexpr int kMaxAlgorithms = 64;
  cublasLtMatmulHeuristicResult_t candidates[kMaxAlgorithms];
  int candidate_count = 0;
  check_lt(cublasLtMatmulAlgoGetHeuristic(
      g_lt_handle, desc.op, desc.a, desc.b, desc.c, desc.c,
      preference, kMaxAlgorithms, candidates, &candidate_count), "cublasLtMatmulAlgoGetHeuristic");
  cublasLtMatmulPreferenceDestroy(preference);
  TORCH_CHECK(candidate_count > 0, "no cuBLASLt candidates for shape");

  auto output = torch::empty({m, n}, x.options());
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  C10_CUDA_CHECK(cudaEventCreate(&start));
  C10_CUDA_CHECK(cudaEventCreate(&stop));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float best_ms = std::numeric_limits<float>::infinity();
  int best = -1;

  for (int index = 0; index < candidate_count; ++index) {
    auto& candidate = candidates[index];
    if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > kLtWorkspaceBytes) continue;
    cublasStatus_t warmup = run_lt(
        x, weight_t, output, workspace, &candidate.algo, candidate.workspaceSize, desc);
    if (warmup != CUBLAS_STATUS_SUCCESS) continue;
    C10_CUDA_CHECK(cudaEventRecord(start, stream));
    bool valid = true;
    constexpr int kRepetitions = 8;
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
      if (run_lt(x, weight_t, output, workspace, &candidate.algo,
                 candidate.workspaceSize, desc) != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
    }
    C10_CUDA_CHECK(cudaEventRecord(stop, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(stop));
    if (!valid) continue;
    float elapsed_ms = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms < best_ms) {
      best_ms = elapsed_ms;
      best = index;
    }
  }

  C10_CUDA_CHECK(cudaEventDestroy(stop));
  C10_CUDA_CHECK(cudaEventDestroy(start));
  TORCH_CHECK(best >= 0, "no runnable cuBLASLt candidate for shape");
  g_lt_choices[lt_key(m, n, k)] = LtChoice{
      candidates[best].algo, candidates[best].workspaceSize};
}

torch::Tensor lt_mm(torch::Tensor x, torch::Tensor weight_t, torch::Tensor workspace) {
  ensure_lt_handle();
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(weight_t.size(1));
  auto found = g_lt_choices.find(lt_key(m, n, k));
  TORCH_CHECK(found != g_lt_choices.end(), "cuBLASLt shape was not tuned in prepare");
  auto output = torch::empty({m, n}, x.options());
  LtDescriptors desc(m, n, k);
  check_lt(run_lt(x, weight_t, output, workspace, &found->second.algo,
                  found->second.workspace_bytes, desc), "cublasLtMatmul");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("router_fp32_gemv", &router_fp32_gemv);
  module.def("lt_tune", &lt_tune);
  module.def("lt_mm", &lt_mm);
}
"""


_router_extension = load_inline(
    name="okc_router_fp32_gemv_v1",
    cpp_sources="",
    cuda_sources=_ROUTER_CUDA,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    with_cuda=True,
    verbose=False,
)


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


def prepare(weights: Weights):
    """Pre-split the FP32 router exactly as SGLang does for larger batches."""
    state = {name: weight.contiguous() for name, weight in weights.items()}
    for name, weight in weights.items():
        if weight.dtype == torch.bfloat16:
            state[f"{name}_t"] = weight.t().contiguous()
    state["unpooler_slice_t"] = weights["unpooler"][:1024].t().contiguous()
    router_hi = weights["router"].to(torch.bfloat16)
    router_lo = (weights["router"] - router_hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = router_hi.t().contiguous()
    state["router_lo_t"] = router_lo.t().contiguous()
    # Pack both BF16 residual components into a single wider GEMM.  This keeps
    # exactly the same hi+lo representation while paying one tensor-core GEMM
    # launch instead of two; the two FP32 result halves are reduced afterward.
    state["router_pair_t"] = torch.cat(
        (state["router_hi_t"], state["router_lo_t"]), dim=1
    ).contiguous()
    state["lt_workspace"] = torch.empty(32 << 20, dtype=torch.uint8, device=router_hi.device)
    state["unpooler_m8_out"] = torch.empty(
        (8, 1024), dtype=torch.bfloat16, device=router_hi.device
    )
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        for m in (8, 16, 32, 128, 256):
            if layer == "unpooler" and m == 8:
                weight_t = state["unpooler_slice_t"]
            else:
                weight_t = state[f"{layer}_t"]
            sample = torch.empty((m, weight_t.shape[0]), dtype=torch.bfloat16, device=weight_t.device)
            _router_extension.lt_tune(sample, weight_t, state["lt_workspace"])
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """Return x @ W[:n].T, where W is weights[config["layer"]]: bfloat16, or float32 for the router."""
    x = data["x"]
    layer = data["config"]["layer"]
    n = data["config"]["n"]

    if layer != "router":
        weight = state[layer][:n]
        # This JIT kernel is shipped in the judge's SGLang image and is tuned
        # specifically for Hopper's single-token, weight-streaming workload.
        if x.shape[0] == 1:
            return hopper_bf16_gemv(x, weight)
        # TinyGEMM is strongly positive only for this narrow-output verify
        # projection.  Its m=8 O projection was substantially slower, while
        # qkv/pooler gains were absent or within run-to-run noise.
        if layer == "unpooler" and x.shape[0] == 8:
            output = state["unpooler_m8_out"]
            tinygemm_bf16(x, weight, output)
            return output
        # The cold-L2 profile shows that SGLang's native dispatcher has better
        # tactics for these shapes than our prepare-time cuBLASLt autotuner.
        # Keep the tuned Lt path only where it demonstrated a real win.
        if ((layer == "o")
                or (layer == "unpooler" and x.shape[0] == 8)
                or (layer == "qkv_gate" and x.shape[0] in (16, 32, 256))):
            return _bf16_gemm_dispatch_impl(x, weight, None)
        if layer == "unpooler" and n == 1024:
            weight_t = state["unpooler_slice_t"]
        else:
            weight_t = state[f"{layer}_t"]
        if x.shape[0] in (8, 16, 32, 128, 256):
            return _router_extension.lt_mm(x, weight_t, state["lt_workspace"])
        return torch.mm(x, weight_t)

    if x.shape[0] <= 8:
        return _router_extension.router_fp32_gemv(x, state["router"])
    pair = torch.mm(x, state["router_pair_t"], out_dtype=torch.float32)
    return pair[:, :256] + pair[:, 256:]
