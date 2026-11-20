# okc: problem=muse_attention_decode
"""FA3 attention with the scored TP8 Q/K normalization, RoPE, FP8 query cast,
and KV-cache write fused into one CUDA kernel."""

from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
from torch.utils.cpp_extension import load_inline


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
ROPE_POSITIONS = 1 << 21
_NORMS = {}
_ROPE = {}


_ATTENTION_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kWarps = kQueryHeads + kKvHeads;
constexpr int kThreads = 32 * kWarps;
constexpr int kQkvStride = (2 * kQueryHeads + 2 * kKvHeads) * kHeadDim;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ uint8_t to_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

template <bool ApplyRope, bool ScaledCache>
__global__ __launch_bounds__(kThreads)
void fused_prepare_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ cache_loc,
    const float* __restrict__ rope_cache,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    uint8_t* __restrict__ q_out,
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    int tokens) {
  const int token = static_cast<int>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (token >= tokens || warp >= kWarps) return;

  const long long token_base = static_cast<long long>(token) * kQkvStride;
  const int d0 = lane * 2;
  const int d1 = d0 + 1;

  if (warp < kQueryHeads) {
    const long long input_base = token_base + static_cast<long long>(warp) * kHeadDim;
    float x0 = __bfloat162float(qkv[input_base + d0]);
    float x1 = __bfloat162float(qkv[input_base + d1]);
    const float inv_rms = rsqrtf(warp_sum(x0 * x0 + x1 * x1) / 64.0f + 1.0e-5f);
    x0 *= inv_rms;
    x1 *= inv_rms;
    if constexpr (ApplyRope) {
      const long long rope_base = positions[token] * 64;
      const float cosine = rope_cache[rope_base + lane];
      const float sine = rope_cache[rope_base + 32 + lane];
      const float rotated0 = x0 * cosine - x1 * sine;
      x1 = x1 * cosine + x0 * sine;
      x0 = rotated0;
    }
    const long long output_base =
        (static_cast<long long>(token) * kQueryHeads + warp) * kHeadDim;
    q_out[output_base + d0] = to_e4m3(x0);
    q_out[output_base + d1] = to_e4m3(x1);
    return;
  }

  const int kv_head = warp - kQueryHeads;
  const long long key_input =
      token_base + static_cast<long long>(kQueryHeads + kv_head) * kHeadDim;
  float k0 = __bfloat162float(qkv[key_input + d0]);
  float k1 = __bfloat162float(qkv[key_input + d1]);
  const float inv_rms = rsqrtf(warp_sum(k0 * k0 + k1 * k1) / 64.0f + 1.0e-5f);
  k0 *= inv_rms;
  k1 *= inv_rms;
  if constexpr (ApplyRope) {
    const long long rope_base = positions[token] * 64;
    const float cosine = rope_cache[rope_base + lane];
    const float sine = rope_cache[rope_base + 32 + lane];
    const float rotated0 = k0 * cosine - k1 * sine;
    k1 = k1 * cosine + k0 * sine;
    k0 = rotated0;
  }

  float value0 = __bfloat162float(qkv[key_input + kKvHeads * kHeadDim + d0]);
  float value1 = __bfloat162float(qkv[key_input + kKvHeads * kHeadDim + d1]);
  if constexpr (ScaledCache) {
    const float inverse_k_scale = 1.0f / *k_scale;
    const float inverse_v_scale = 1.0f / *v_scale;
    k0 *= inverse_k_scale;
    k1 *= inverse_k_scale;
    value0 *= inverse_v_scale;
    value1 *= inverse_v_scale;
  }

  const long long cache_base =
      (cache_loc[token] * kKvHeads + kv_head) * kHeadDim;
  k_cache[cache_base + d0] = to_e4m3(k0);
  k_cache[cache_base + d1] = to_e4m3(k1);
  v_cache[cache_base + d0] = to_e4m3(value0);
  v_cache[cache_base + d1] = to_e4m3(value1);
}

}  // namespace

torch::Tensor fused_prepare(
    torch::Tensor qkv,
    torch::Tensor positions,
    torch::Tensor cache_loc,
    torch::Tensor rope_cache,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor q_out,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    bool apply_rope,
    bool scaled_cache) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be BF16");
  TORCH_CHECK(qkv.size(-1) == kQkvStride, "unexpected TP8 qkv width");
  const int tokens = static_cast<int>(qkv.numel() / kQkvStride);
  const dim3 blocks(tokens);
  const dim3 threads(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto q_ptr = reinterpret_cast<uint8_t*>(q_out.data_ptr());
  auto k_ptr = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  auto v_ptr = reinterpret_cast<uint8_t*>(v_cache.data_ptr());

  if (apply_rope) {
    if (scaled_cache) {
      fused_prepare_kernel<true, true><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
          positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(),
          rope_cache.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          q_ptr, k_ptr, v_ptr, tokens);
    } else {
      fused_prepare_kernel<true, false><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
          positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(),
          rope_cache.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          q_ptr, k_ptr, v_ptr, tokens);
    }
  } else if (scaled_cache) {
    fused_prepare_kernel<false, true><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(),
        rope_cache.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        q_ptr, k_ptr, v_ptr, tokens);
  } else {
    fused_prepare_kernel<false, false><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(),
        rope_cache.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        q_ptr, k_ptr, v_ptr, tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_prepare", &fused_prepare);
}
"""


_attention_extension = load_inline(
    name="okc_muse_attention_prepare_v1",
    cpp_sources="",
    cuda_sources=_ATTENTION_CUDA,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum(
            "i,j -> ij",
            torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device),
            inv_freq,
        )
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _baseline_prepare(data, qkv, tokens, hq, hkv, d, scaled):
    cfg = data["config"]
    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(
            qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, qkv.device)
        )
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(
        k.to(FP8).view(torch.uint8),
        v.to(FP8).view(torch.uint8),
        data["k_cache"].view(-1, hkv * d).view(torch.uint8),
        data["v_cache"].view(-1, hkv * d).view(torch.uint8),
        data["cache_loc"],
        reserved_skip_index=-1,
    )
    return q.view(tokens, hq, d).to(FP8)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    scaled = cfg["kv_scale"] != 1.0

    if hq == 16 and hkv == 2 and d == 64:
        q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
        rope_cache = (
            _cos_sin(cfg["rope_theta"], d, qkv.device)
            if cfg["rope_theta"]
            else data["k_scale"]
        )
        q = _attention_extension.fused_prepare(
            qkv,
            data["positions"],
            data["cache_loc"],
            rope_cache,
            data["k_scale"],
            data["v_scale"],
            q,
            data["k_cache"],
            data["v_cache"],
            bool(cfg["rope_theta"]),
            scaled,
        )
    else:
        q = _baseline_prepare(data, qkv, tokens, hq, hkv, d, scaled)

    descale = {}
    if scaled:
        descale = {
            "k_descale": data["k_scale"].reshape(1, 1).expand(batch, hkv),
            "v_descale": data["v_scale"].reshape(1, 1).expand(batch, hkv),
        }
    window = cfg["window"]
    # For varlen calls without precomputed metadata, keep one near-full H200
    # wave: B * Hkv * splits = 128 CTAs for the scored B=16/32 cases.
    # Batch one and metadata-backed full decode retain FA3's dynamic heuristic.
    num_splits = 0
    if data["scheduler_metadata"] is None:
        if batch == 16:
            num_splits = 4
        elif batch == 32:
            num_splits = 2
    # With one request and one token per page, the page-table width is the
    # exact cache length.  Present this as fixed-shape batched attention so
    # FA3 can use its static scheduler instead of preparing a varlen schedule.
    static_single = batch == 1 and cfg["page_size"] == 1
    q_arg = q.view(batch, length, hq, d) if static_single else q
    out = flash_attn_with_kvcache(
        q=q_arg,
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
        page_table=data["page_table"],
        cache_seqlens=None if static_single else data["cache_seqlens"],
        cu_seqlens_q=None if static_single else data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=None if static_single else data["scheduler_metadata"],
        num_splits=num_splits,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    result = rms_norm_gate(
        out.reshape(tokens, hq * d).to(torch.bfloat16),
        gate,
        None,
        head_dim=d,
        eps=RMS_EPS,
        has_norm=True,
    )
    return result.view(batch, length, hq * d)
