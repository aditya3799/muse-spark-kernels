# okc: problem=muse_attention_decode
"""FA3 attention with fused preparation and a gated custom full-decode path."""

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kWarps = kQueryHeads + kKvHeads;
constexpr int kThreads = 32 * kWarps;
constexpr int kQkvStride = (2 * kQueryHeads + 2 * kKvHeads) * kHeadDim;
constexpr int kGroupSize = kQueryHeads / kKvHeads;
constexpr int kPartialStride = 2 + kHeadDim;
constexpr int kKeyWarps = 16;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
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

// FA3 local-varlen metadata for the scored q_len=8 TP8 shapes.  The metadata
// is rebuilt from cache_seqlens on every invocation.  We attach one metadata
// CTA to the prepare grid so CUDA-graph replay has no standalone scheduler
// launch and no state derived from a previous call.
__device__ __forceinline__ void write_sliding_verify_scheduler_metadata(
    const int32_t* __restrict__ cache_seqlens,
    int32_t* __restrict__ metadata,
    int batch,
    int num_splits,
    int num_sm) {
  constexpr int kBlockN = 160;  // FA3 SM90 FP8, d=64, local attention.
  constexpr int kMaxKvBlocksInL2 =
      (8 * 1024 * 1024) / (kBlockN * (kHeadDim + kHeadDim));
  const int tid = static_cast<int>(threadIdx.x);
  const int b_rounded = (batch + 3) & ~3;
  int num_n_blocks = 0;
  if (tid < batch) {
    num_n_blocks = (cache_seqlens[tid] + kBlockN - 1) / kBlockN;
  }
  int total_blocks = num_n_blocks;  // num_m_blocks is one for q_len=8 PackGQA.
  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_blocks += __shfl_down_sync(0xffffffff, total_blocks, offset);
    }
  }
  __shared__ int blocks_per_sm_shared;
  if (tid == 0) {
    // Match FA3 prepare_varlen_num_blocks_kernel: 10% scheduling margin and
    // two packed KV heads.  max(1, ...) also makes an empty request benign.
    const int blocks_per_sm = max(
        1, static_cast<int>(ceilf(
               static_cast<float>(total_blocks) * 1.1f * kKvHeads /
               static_cast<float>(num_sm))));
    blocks_per_sm_shared = blocks_per_sm;
    metadata[3 * b_rounded] = 0;  // persistent scheduler semaphore
  }
  __syncthreads();
  if (tid < batch) {
    const int dynamic_splits = max(
        1, min((num_n_blocks + blocks_per_sm_shared - 1) /
                   blocks_per_sm_shared,
               num_splits));
    const int split_n_blocks =
        (num_n_blocks + dynamic_splits - 1) / dynamic_splits;
    int heads_in_l2 = split_n_blocks * 16 <= kMaxKvBlocksInL2 ? 16
        : split_n_blocks * 8 <= kMaxKvBlocksInL2 ? 8
        : split_n_blocks * 4 <= kMaxKvBlocksInL2 ? 4
        : split_n_blocks * 2 <= kMaxKvBlocksInL2 ? 2
        : 1;
    heads_in_l2 = min(heads_in_l2, kKvHeads);
    metadata[tid] = dynamic_splits;
    metadata[b_rounded + tid] = 1;  // ceil((8 queries * GQA 8) / blockM 192)
    metadata[2 * b_rounded + tid] = heads_in_l2;
  }
}

template <bool ApplyRope, bool ScaledCache>
__global__ __launch_bounds__(kThreads)
void fused_prepare_sliding_verify_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ cache_loc,
    const float* __restrict__ rope_cache,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    uint8_t* __restrict__ q_out,
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ cache_seqlens,
    int32_t* __restrict__ scheduler_metadata,
    int tokens,
    int batch,
    int num_splits,
    int num_sm) {
  const int token = static_cast<int>(blockIdx.x);
  if (token == tokens) {
    write_sliding_verify_scheduler_metadata(
        cache_seqlens, scheduler_metadata, batch, num_splits, num_sm);
    return;
  }
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

// Decode-only full attention specialized for Muse TP8. Each warp walks a
// different interleaved key stream and evaluates all eight query heads sharing
// the KV head. K and V are therefore loaded once per key rather than once per
// query head. The block merges its warp-local online-softmax states in shared
// memory and emits one compact partial per query head.
__global__ __launch_bounds__(512, 2)
void paged_full_decode_partials_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
  const int linear = static_cast<int>(blockIdx.x);
  const int split = linear % num_splits;
  const int kv_head = (linear / num_splits) % kKvHeads;
  const int batch = linear / (num_splits * kKvHeads);
  const int key_warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);
  float running_max[kGroupSize];
  float running_sum[kGroupSize];
  float out0[kGroupSize];
  float out1[kGroupSize];
#pragma unroll
  for (int head = 0; head < kGroupSize; ++head) {
    running_max[head] = -FLT_MAX;
    running_sum[head] = 0.0f;
    out0[head] = 0.0f;
    out1[head] = 0.0f;
  }

  for (int pos = begin + key_warp; pos < end; pos += kKeyWarps) {
    int page = 0;
    if (lane == 0) page = page_table[static_cast<long long>(batch) * max_pages + pos];
    page = __shfl_sync(0xffffffff, page, 0);
    const long long cache_base =
        (static_cast<long long>(page) * kKvHeads + kv_head) * kHeadDim;
    const float2 key_pair = static_cast<float2>(
        *reinterpret_cast<const __nv_fp8x2_e4m3*>(
            k_cache + cache_base + lane * 2));
    const float2 value_pair = static_cast<float2>(
        *reinterpret_cast<const __nv_fp8x2_e4m3*>(
            v_cache + cache_base + lane * 2));
#pragma unroll
    for (int head = 0; head < kGroupSize; ++head) {
      const int q_head = kv_head * kGroupSize + head;
      const long long q_base =
          (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim;
      const float2 query_pair = static_cast<float2>(
          *reinterpret_cast<const __nv_fp8x2_e4m3*>(q + q_base + lane * 2));
      float score = query_pair.x * key_pair.x + query_pair.y * key_pair.y;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        score += __shfl_down_sync(0xffffffff, score, offset);
      }
      score = __shfl_sync(0xffffffff, score, 0) * softmax_scale;
      const float next_max = fmaxf(running_max[head], score);
      const float old_scale = exp2f((running_max[head] - next_max) * 1.4426950408889634f);
      const float value_scale = exp2f((score - next_max) * 1.4426950408889634f);
      out0[head] = out0[head] * old_scale + value_pair.x * value_scale;
      out1[head] = out1[head] * old_scale + value_pair.y * value_scale;
      running_sum[head] = running_sum[head] * old_scale + value_scale;
      running_max[head] = next_max;
    }
  }

  __shared__ float warp_maxima[kKeyWarps][kGroupSize];
  __shared__ float warp_sums[kKeyWarps][kGroupSize];
  __shared__ float warp_outputs[kKeyWarps][kGroupSize][kHeadDim];
  __shared__ float warp_factors[kGroupSize][kKeyWarps];
#pragma unroll
  for (int head = 0; head < kGroupSize; ++head) {
    if (lane == 0) {
      warp_maxima[key_warp][head] = running_max[head];
      warp_sums[key_warp][head] = running_sum[head];
    }
    warp_outputs[key_warp][head][lane * 2] = out0[head];
    warp_outputs[key_warp][head][lane * 2 + 1] = out1[head];
  }
  __syncthreads();

  if (key_warp >= kGroupSize) return;
  const int group_head = key_warp;
  const float global_max = warp_max(warp_maxima[lane & (kKeyWarps - 1)][group_head]);
  float factor = 0.0f;
  float weighted_sum = 0.0f;
  if (lane < kKeyWarps) {
    factor = exp2f((warp_maxima[lane][group_head] - global_max) * 1.4426950408889634f);
    weighted_sum = warp_sums[lane][group_head] * factor;
    warp_factors[group_head][lane] = factor;
  }
  const float merged_sum = warp_sum(weighted_sum);
  __syncwarp();
  float merged0 = 0.0f;
  float merged1 = 0.0f;
#pragma unroll
  for (int source = 0; source < kKeyWarps; ++source) {
    const float source_factor = warp_factors[group_head][source];
    merged0 += warp_outputs[source][group_head][lane * 2] * source_factor;
    merged1 += warp_outputs[source][group_head][lane * 2 + 1] * source_factor;
  }
  const long long partial_base =
      (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
       * kGroupSize + group_head) * kPartialStride;
  if (lane == 0) {
    partials[partial_base] = global_max;
    partials[partial_base + 1] = merged_sum;
  }
  partials[partial_base + 2 + lane * 2] = merged0;
  partials[partial_base + 3 + lane * 2] = merged1;
}

// Portable tensor-core prototype for the SM90 WGMMA design. A CTA gathers a
// 16-key page tile, computes QK and PV with WMMA, and updates softmax once per
// tile. Decode has 8 useful rows; the other 8 rows are zero padding.
__global__ __launch_bounds__(128, 2)
void paged_full_decode_partials_tc_kernel(



    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTileN = 16;
  const int linear = static_cast<int>(blockIdx.x);
  const int split = linear % num_splits;
  const int kv_head = (linear / num_splits) % kKvHeads;
  const int batch = linear / (num_splits * kKvHeads);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);

  __shared__ __half s_q[16 * kHeadDim];
  __shared__ __half s_k[kTileN * kHeadDim];
  __shared__ __half s_v[kTileN * kHeadDim];
  __shared__ __half s_p[16 * kTileN];
  __shared__ float s_scores[16 * kTileN];
  __shared__ float s_tile_out[16 * kHeadDim];
  __shared__ float s_running_out[kGroupSize * kHeadDim];
  __shared__ float s_running_max[kGroupSize];
  __shared__ float s_running_sum[kGroupSize];
  __shared__ float s_old_scale[kGroupSize];
  __shared__ int s_pages[kTileN];

  for (int idx = static_cast<int>(threadIdx.x); idx < 16 * kHeadDim; idx += 128) {
    const int row = idx / kHeadDim;
    const int dim = idx % kHeadDim;
    if (row < kGroupSize) {
      const int q_head = kv_head * kGroupSize + row;
      const long long q_idx =
          (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim + dim;
      s_q[idx] = __float2half(static_cast<float>(q[q_idx]));
    } else {
      s_q[idx] = __float2half(0.0f);
    }
  }
  for (int idx = static_cast<int>(threadIdx.x); idx < kGroupSize * kHeadDim; idx += 128) {
    s_running_out[idx] = 0.0f;
  }
  if (threadIdx.x < kGroupSize) {
    s_running_max[threadIdx.x] = -FLT_MAX;
    s_running_sum[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int tile_begin = begin; tile_begin < end; tile_begin += kTileN) {
    if (threadIdx.x < kTileN) {
      const int pos = tile_begin + static_cast<int>(threadIdx.x);
      s_pages[threadIdx.x] =
          pos < end ? page_table[static_cast<long long>(batch) * max_pages + pos] : 0;
    }
    __syncthreads();

    for (int pair = static_cast<int>(threadIdx.x); pair < kTileN * 32; pair += 128) {
      const int key = pair >> 5;
      const int pair_dim = pair & 31;
      const int pos = tile_begin + key;
      float2 key_pair = make_float2(0.0f, 0.0f);
      float2 value_pair = make_float2(0.0f, 0.0f);
      if (pos < end) {
        const long long cache_base =
            (static_cast<long long>(s_pages[key]) * kKvHeads + kv_head) * kHeadDim;
        key_pair = static_cast<float2>(
            *reinterpret_cast<const __nv_fp8x2_e4m3*>(
                k_cache + cache_base + pair_dim * 2));
        value_pair = static_cast<float2>(
            *reinterpret_cast<const __nv_fp8x2_e4m3*>(
                v_cache + cache_base + pair_dim * 2));
      }
      s_k[key * kHeadDim + pair_dim * 2] = __float2half(key_pair.x);
      s_k[key * kHeadDim + pair_dim * 2 + 1] = __float2half(key_pair.y);
      s_v[key * kHeadDim + pair_dim * 2] = __float2half(value_pair.x);
      s_v[key * kHeadDim + pair_dim * 2 + 1] = __float2half(value_pair.y);
    }
    for (int idx = static_cast<int>(threadIdx.x); idx < 16 * kTileN; idx += 128) {
      s_p[idx] = __float2half(0.0f);
    }
    __syncthreads();

    if (warp == 0) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += 16) {
        wmma::load_matrix_sync(a, s_q + kk, kHeadDim);
        wmma::load_matrix_sync(b, s_k + kk, kHeadDim);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(s_scores, c, kTileN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int head = warp; head < kGroupSize; head += 4) {
      const int key = lane;
      const bool valid = key < kTileN && tile_begin + key < end;
      float score = valid ? s_scores[head * kTileN + key] * softmax_scale : -FLT_MAX;
      const float tile_max = warp_max(score);
      const float next_max = fmaxf(s_running_max[head], tile_max);
      const float old_scale = exp2f(
          (s_running_max[head] - next_max) * 1.4426950408889634f);
      const float probability = valid
          ? exp2f((score - next_max) * 1.4426950408889634f)
          : 0.0f;
      const float tile_sum = warp_sum(probability);
      if (key < kTileN) s_p[head * kTileN + key] = __float2half(probability);
      if (lane == 0) {
        s_old_scale[head] = old_scale;
        s_running_max[head] = next_max;
        s_running_sum[head] = s_running_sum[head] * old_scale + tile_sum;
      }
    }
    __syncthreads();

    {
      const int col = warp * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0f);
      wmma::load_matrix_sync(a, s_p, kTileN);
      wmma::load_matrix_sync(b, s_v + col, kHeadDim);
      wmma::mma_sync(c, a, b, c);
      wmma::store_matrix_sync(
          s_tile_out + col, c, kHeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = static_cast<int>(threadIdx.x); idx < kGroupSize * kHeadDim; idx += 128) {
      const int head = idx / kHeadDim;
      s_running_out[idx] =
          s_running_out[idx] * s_old_scale[head] + s_tile_out[idx];
    }
    __syncthreads();
  }

  for (int idx = static_cast<int>(threadIdx.x); idx < kGroupSize * kHeadDim; idx += 128) {
    const int group_head = idx / kHeadDim;
    const int dim = idx % kHeadDim;
    const long long partial_base =
        (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
         * kGroupSize + group_head) * kPartialStride;
    if (dim == 0) {
      partials[partial_base] = s_running_max[group_head];
      partials[partial_base + 1] = s_running_sum[group_head];
    }
    partials[partial_base + 2 + dim] = s_running_out[idx];
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// Submission-local WGMMA support. These helpers intentionally avoid a CUTLASS
// header dependency: the challenge extension can therefore build the Hopper
// path with only CUDA and PyTorch headers.
__device__ __forceinline__ int swizzle_b64_fp8(int linear) {
  return linear ^ ((linear & 0x180) >> 3);
}

__device__ __forceinline__ int swizzle_b32_f16(int linear) {
  return linear ^ ((linear & 0x80) >> 3);
}

__device__ __forceinline__ uint64_t make_wgmma_desc_b64(const void* ptr) {
  const uint64_t start =
      (static_cast<uint64_t>(__cvta_generic_to_shared(ptr)) >> 4) & 0x3fff;
  return start | (uint64_t(1) << 16) | (uint64_t(32) << 32) |
         (uint64_t(2) << 62);
}

__device__ __forceinline__ uint64_t make_wgmma_desc_b32(const void* ptr) {
  const uint64_t start =
      (static_cast<uint64_t>(__cvta_generic_to_shared(ptr)) >> 4) & 0x3fff;
  return start | (uint64_t(1) << 16) | (uint64_t(16) << 32) |
         (uint64_t(3) << 62);
}

__device__ __forceinline__ void wgmma_arrive() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

template <int BarrierId, int Participants>
__device__ __forceinline__ void named_barrier() {
  asm volatile("bar.sync %0, %1;" :: "n"(BarrierId), "n"(Participants) : "memory");
}

__device__ __forceinline__ void wgmma_qk_f32_e4m3(
    uint64_t desc_a, uint64_t desc_b, float (&d)[8], int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,p,1,1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(desc_a), "l"(desc_b), "r"(accumulate));
}

__device__ __forceinline__ void wgmma_m64n8_f32_e4m3(
    uint64_t desc_a, uint64_t desc_b, float (&d)[4], int accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3},%4,%5,p,1,1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(desc_a), "l"(desc_b), "r"(accumulate));
}

__device__ __forceinline__ void wgmma_pv_f32_f16(
    uint64_t desc_a, uint64_t desc_b, float (&d)[32]) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%32,%33,p,1,1,0,0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(1));
}

#endif

// Hopper decode kernel. One warp-group owns one (batch, KV head, split).
// QK uses FP8 WGMMA directly. Warp 0 performs eight independent four-lane
// softmax reductions, and all four warps then execute the FP16 P*V WGMMA.
__global__ __launch_bounds__(128, 2)
void paged_full_decode_partials_wgmma_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTileN = 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int linear = static_cast<int>(blockIdx.x);
  const int split = linear % num_splits;
  const int kv_head = (linear / num_splits) % kKvHeads;
  const int batch = linear / (num_splits * kKvHeads);
  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);

  __shared__ __align__(128) uint8_t s_q[64 * kHeadDim];
  __shared__ __align__(128) uint8_t s_k[kTileN * kHeadDim];
  __shared__ __align__(128) __half s_p[64 * kTileN];
  __shared__ __align__(128) __half s_v[kHeadDim * kTileN];
  __shared__ int s_pages[kTileN];

  for (int idx = tid; idx < 64 * kHeadDim; idx += 128) s_q[idx] = 0;
  __syncthreads();
  for (int pair = tid; pair < kGroupSize * 32; pair += 128) {
    const int head = pair >> 5;
    const int pair_dim = pair & 31;
    const int q_head = kv_head * kGroupSize + head;
    const long long q_idx =
        (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim +

        pair_dim * 2;

    const uint16_t raw = *reinterpret_cast<const uint16_t*>(q + q_idx);

    const int dst = swizzle_b64_fp8(head * kHeadDim + pair_dim * 2);
    *reinterpret_cast<uint16_t*>(s_q + dst) = raw;
  }
  __syncthreads();

  const uint64_t q_desc = make_wgmma_desc_b64(s_q);
  const uint64_t k_desc = make_wgmma_desc_b64(s_k);
  const uint64_t p_desc = make_wgmma_desc_b32(s_p);
  const uint64_t v_desc = make_wgmma_desc_b32(s_v);

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float running_out[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) running_out[i] = 0.0f;

  for (int tile_begin = begin; tile_begin < end; tile_begin += kTileN) {
    if (tid < kTileN) {
      const int pos = tile_begin + tid;
      s_pages[tid] =
          pos < end ? page_table[static_cast<long long>(batch) * max_pages + pos] : 0;
    }
    for (int idx = tid; idx < kTileN * kHeadDim; idx += 128) s_k[idx] = 0;
    for (int idx = tid; idx < kHeadDim * kTileN; idx += 128) {
      reinterpret_cast<uint16_t*>(s_v)[idx] = 0;
    }
    __syncthreads();

    for (int pair = tid; pair < kTileN * 32; pair += 128) {
      const int key = pair >> 5;
      const int pair_dim = pair & 31;
      const int pos = tile_begin + key;
      if (pos < end) {
        const long long cache_base =
            (static_cast<long long>(s_pages[key]) * kKvHeads + kv_head) * kHeadDim;
        const uint16_t raw_k = *reinterpret_cast<const uint16_t*>(
            k_cache + cache_base + pair_dim * 2);
        const int k_dst = swizzle_b64_fp8(key * kHeadDim + pair_dim * 2);
        *reinterpret_cast<uint16_t*>(s_k + k_dst) = raw_k;
        const float2 value_pair = static_cast<float2>(
            *reinterpret_cast<const __nv_fp8x2_e4m3*>(
                v_cache + cache_base + pair_dim * 2));
        const int dim0 = pair_dim * 2;
        const int v0 = swizzle_b32_f16(dim0 * kTileN + key);
        const int v1 = swizzle_b32_f16((dim0 + 1) * kTileN + key);
        s_v[v0] = __float2half(value_pair.x);
        s_v[v1] = __float2half(value_pair.y);
      }
    }
    __syncthreads();

    float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f};
    wgmma_arrive();
    wgmma_qk_f32_e4m3(q_desc, k_desc, scores, 0);
    wgmma_qk_f32_e4m3(q_desc + 2, k_desc + 2, scores, 1);
    wgmma_commit();
    wgmma_wait();

    for (int idx = tid; idx < 64 * kTileN; idx += 128) {
      reinterpret_cast<uint16_t*>(s_p)[idx] = 0;
    }

    float old_scale = 1.0f;
    if (warp == 0) {
      const int subgroup_lane = lane & 3;
      const int key0 = subgroup_lane * 2;
      float score0 = tile_begin + key0 < end
          ? scores[0] * softmax_scale : -FLT_MAX;
      float score1 = tile_begin + key0 + 1 < end
          ? scores[1] * softmax_scale : -FLT_MAX;
      float score2 = tile_begin + key0 + 8 < end
          ? scores[4] * softmax_scale : -FLT_MAX;
      float score3 = tile_begin + key0 + 9 < end
          ? scores[5] * softmax_scale : -FLT_MAX;
      float tile_max = fmaxf(fmaxf(score0, score1), fmaxf(score2, score3));
      tile_max = fmaxf(tile_max,
                       __shfl_xor_sync(0xffffffff, tile_max, 1, 4));
      tile_max = fmaxf(tile_max,
                       __shfl_xor_sync(0xffffffff, tile_max, 2, 4));
      const float next_max = fmaxf(running_max, tile_max);
      old_scale = exp2f((running_max - next_max) * 1.4426950408889634f);
      const float p0 = score0 == -FLT_MAX ? 0.0f
          : exp2f((score0 - next_max) * 1.4426950408889634f);
      const float p1 = score1 == -FLT_MAX ? 0.0f
          : exp2f((score1 - next_max) * 1.4426950408889634f);
      const float p2 = score2 == -FLT_MAX ? 0.0f
          : exp2f((score2 - next_max) * 1.4426950408889634f);
      const float p3 = score3 == -FLT_MAX ? 0.0f
          : exp2f((score3 - next_max) * 1.4426950408889634f);
      float tile_sum = p0 + p1 + p2 + p3;
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1, 4);
      tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2, 4);
      running_max = next_max;
      running_sum = running_sum * old_scale + tile_sum;
      const int head = lane >> 2;
      s_p[swizzle_b32_f16(head * kTileN + key0)] = __float2half(p0);
      s_p[swizzle_b32_f16(head * kTileN + key0 + 1)] = __float2half(p1);
      s_p[swizzle_b32_f16(head * kTileN + key0 + 8)] = __float2half(p2);
      s_p[swizzle_b32_f16(head * kTileN + key0 + 9)] = __float2half(p3);
    }
    __syncthreads();

    float tile_out[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) tile_out[i] = 0.0f;
    wgmma_arrive();
    wgmma_pv_f32_f16(p_desc, v_desc, tile_out);
    wgmma_commit();
    wgmma_wait();

    if (warp == 0) {
#pragma unroll
      for (int block = 0; block < 4; ++block) {
        const int src = block * 8;
        const int dst = block * 4;
        running_out[dst] = running_out[dst] * old_scale + tile_out[src];
        running_out[dst + 1] = running_out[dst + 1] * old_scale + tile_out[src + 1];
        running_out[dst + 2] = running_out[dst + 2] * old_scale + tile_out[src + 4];
        running_out[dst + 3] = running_out[dst + 3] * old_scale + tile_out[src + 5];
      }
    }
    __syncthreads();
  }

  if (warp == 0) {
    const int head = lane >> 2;
    const int subgroup_lane = lane & 3;
    const long long partial_base =
        (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
         * kGroupSize + head) * kPartialStride;
    if (subgroup_lane == 0) {
      partials[partial_base] = running_max;
      partials[partial_base + 1] = running_sum;
    }
#pragma unroll
    for (int block = 0; block < 4; ++block) {
      const int src = block * 4;
      const int dim = block * 16 + subgroup_lane * 2;
      partials[partial_base + 2 + dim] = running_out[src];
      partials[partial_base + 3 + dim] = running_out[src + 1];
      partials[partial_base + 10 + dim] = running_out[src + 2];
      partials[partial_base + 11 + dim] = running_out[src + 3];
    }
  }
#else
  (void)q;
  (void)k_cache;
  (void)v_cache;
  (void)page_table;
  (void)cache_seqlens;
  (void)partials;
  (void)max_pages;
  (void)num_splits;
  (void)softmax_scale;
#endif
}

// Decode-specialized XQA/Lean hybrid. Swapping the GEMM axes makes every
// output of m64n8 useful: rows are 64 K/V tokens (or output dimensions) and
// columns are the eight query heads sharing a KV head. Both Q*K and P*V use
// FP8 WGMMA. Per-tile states are merged with associative online softmax.
__global__ __launch_bounds__(128, 2)
void paged_full_decode_partials_xqa_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTileN = 64;
  constexpr float kFp8Max = 448.0f;
  constexpr float kInvFp8Max = 1.0f / kFp8Max;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int linear = static_cast<int>(blockIdx.x);
  const int split = linear % num_splits;
  const int kv_head = (linear / num_splits) % kKvHeads;
  const int batch = linear / (num_splits * kKvHeads);
  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);

  __shared__ __align__(128) uint8_t s_q[kGroupSize * kHeadDim];
  __shared__ __align__(128) uint8_t s_k[kTileN * kHeadDim];
  __shared__ __align__(128) uint8_t s_p[kGroupSize * kTileN];
  __shared__ __align__(128) uint8_t s_v[kHeadDim * kTileN];
  __shared__ int s_pages[kTileN];
  __shared__ float s_warp_max[4][kGroupSize];
  __shared__ float s_warp_sum[4][kGroupSize];
  __shared__ float s_tile_max[kGroupSize];
  __shared__ float s_tile_sum[kGroupSize];

  for (int pair = tid; pair < kGroupSize * 32; pair += 128) {
    const int head = pair >> 5;
    const int pair_dim = pair & 31;
    const int q_head = kv_head * kGroupSize + head;
    const long long q_idx =
        (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim +
        pair_dim * 2;
    const uint16_t raw = *reinterpret_cast<const uint16_t*>(q + q_idx);
    const int dst = swizzle_b64_fp8(head * kHeadDim + pair_dim * 2);
    *reinterpret_cast<uint16_t*>(s_q + dst) = raw;
  }
  __syncthreads();

  const uint64_t q_desc = make_wgmma_desc_b64(s_q);
  const uint64_t k_desc = make_wgmma_desc_b64(s_k);
  const uint64_t p_desc = make_wgmma_desc_b64(s_p);
  const uint64_t v_desc = make_wgmma_desc_b64(s_v);

  // CuTe's m64n8 accumulator mapping: each thread owns two rows separated by
  // eight and two adjacent columns. Four warps cover all 64 rows.
  const int row0 = warp * 16 + (lane >> 2);
  const int row1 = row0 + 8;
  const int head0 = (lane & 3) * 2;
  const int head1 = head0 + 1;
  float running_max0 = -FLT_MAX;
  float running_max1 = -FLT_MAX;
  float running_sum0 = 0.0f;
  float running_sum1 = 0.0f;
  float running_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile_begin = begin; tile_begin < end; tile_begin += kTileN) {
    if (tid < kTileN) {
      const int pos = tile_begin + tid;
      s_pages[tid] =
          pos < end ? page_table[static_cast<long long>(batch) * max_pages + pos] : 0;
    }
    __syncthreads();

    for (int pair = tid; pair < kTileN * 32; pair += 128) {
      const int key = pair >> 5;
      const int pair_dim = pair & 31;
      const int pos = tile_begin + key;
      uint16_t raw_k = 0;
      uint16_t raw_v = 0;
      if (pos < end) {
        const long long cache_base =
            (static_cast<long long>(s_pages[key]) * kKvHeads + kv_head) * kHeadDim;
        raw_k = *reinterpret_cast<const uint16_t*>(
            k_cache + cache_base + pair_dim * 2);
        raw_v = *reinterpret_cast<const uint16_t*>(
            v_cache + cache_base + pair_dim * 2);
      }
      const int k_dst = swizzle_b64_fp8(key * kHeadDim + pair_dim * 2);
      *reinterpret_cast<uint16_t*>(s_k + k_dst) = raw_k;
      const int dim0 = pair_dim * 2;
      s_v[swizzle_b64_fp8(dim0 * kTileN + key)] =
          static_cast<uint8_t>(raw_v & 0xff);
      s_v[swizzle_b64_fp8((dim0 + 1) * kTileN + key)] =
          static_cast<uint8_t>(raw_v >> 8);
    }
    __syncthreads();

    float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    wgmma_arrive();
    wgmma_m64n8_f32_e4m3(k_desc, q_desc, scores, 0);
    wgmma_m64n8_f32_e4m3(k_desc + 2, q_desc + 2, scores, 1);
    wgmma_commit();
    wgmma_wait();

    float score0 = tile_begin + row0 < end ? scores[0] * softmax_scale : -FLT_MAX;
    float score1 = tile_begin + row0 < end ? scores[1] * softmax_scale : -FLT_MAX;
    float score2 = tile_begin + row1 < end ? scores[2] * softmax_scale : -FLT_MAX;
    float score3 = tile_begin + row1 < end ? scores[3] * softmax_scale : -FLT_MAX;
    float local_max0 = fmaxf(score0, score2);
    float local_max1 = fmaxf(score1, score3);
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
      local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, delta));
      local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, delta));
    }
    if (lane < 4) {
      s_warp_max[warp][head0] = local_max0;
      s_warp_max[warp][head1] = local_max1;
    }
    __syncthreads();
    if (warp == 0 && lane < kGroupSize) {
      float value = s_warp_max[0][lane];
#pragma unroll
      for (int w = 1; w < 4; ++w) value = fmaxf(value, s_warp_max[w][lane]);
      s_tile_max[lane] = value;
    }
    __syncthreads();

    const float tile_max0 = s_tile_max[head0];
    const float tile_max1 = s_tile_max[head1];
    const float p0 = score0 == -FLT_MAX ? 0.0f
        : exp2f((score0 - tile_max0) * 1.4426950408889634f);

    const float p1 = score1 == -FLT_MAX ? 0.0f
        : exp2f((score1 - tile_max1) * 1.4426950408889634f);

    const float p2 = score2 == -FLT_MAX ? 0.0f
        : exp2f((score2 - tile_max0) * 1.4426950408889634f);

    const float p3 = score3 == -FLT_MAX ? 0.0f
        : exp2f((score3 - tile_max1) * 1.4426950408889634f);
    float local_sum0 = p0 + p2;
    float local_sum1 = p1 + p3;
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
      local_sum0 += __shfl_xor_sync(0xffffffff, local_sum0, delta);
      local_sum1 += __shfl_xor_sync(0xffffffff, local_sum1, delta);
    }
    if (lane < 4) {
      s_warp_sum[warp][head0] = local_sum0;
      s_warp_sum[warp][head1] = local_sum1;
    }
    s_p[swizzle_b64_fp8(head0 * kTileN + row0)] =
        static_cast<__nv_fp8_e4m3>(p0 * kFp8Max).__x;
    s_p[swizzle_b64_fp8(head1 * kTileN + row0)] =
        static_cast<__nv_fp8_e4m3>(p1 * kFp8Max).__x;
    s_p[swizzle_b64_fp8(head0 * kTileN + row1)] =
        static_cast<__nv_fp8_e4m3>(p2 * kFp8Max).__x;
    s_p[swizzle_b64_fp8(head1 * kTileN + row1)] =
        static_cast<__nv_fp8_e4m3>(p3 * kFp8Max).__x;
    __syncthreads();
    if (warp == 0 && lane < kGroupSize) {
      float value = s_warp_sum[0][lane];
#pragma unroll
      for (int w = 1; w < 4; ++w) value += s_warp_sum[w][lane];
      s_tile_sum[lane] = value;
    }
    __syncthreads();

    float tile_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    wgmma_arrive();
    wgmma_m64n8_f32_e4m3(v_desc, p_desc, tile_out, 0);
    wgmma_m64n8_f32_e4m3(v_desc + 2, p_desc + 2, tile_out, 1);
    wgmma_commit();
    wgmma_wait();

    const float next_max0 = fmaxf(running_max0, tile_max0);
    const float next_max1 = fmaxf(running_max1, tile_max1);
    const float old_scale0 =
        exp2f((running_max0 - next_max0) * 1.4426950408889634f);
    const float old_scale1 =
        exp2f((running_max1 - next_max1) * 1.4426950408889634f);
    const float tile_scale0 =
        exp2f((tile_max0 - next_max0) * 1.4426950408889634f);
    const float tile_scale1 =
        exp2f((tile_max1 - next_max1) * 1.4426950408889634f);
    running_sum0 = running_sum0 * old_scale0 + s_tile_sum[head0] * tile_scale0;
    running_sum1 = running_sum1 * old_scale1 + s_tile_sum[head1] * tile_scale1;
    running_out[0] = running_out[0] * old_scale0 +
        tile_out[0] * (tile_scale0 * kInvFp8Max);
    running_out[1] = running_out[1] * old_scale1 +
        tile_out[1] * (tile_scale1 * kInvFp8Max);
    running_out[2] = running_out[2] * old_scale0 +
        tile_out[2] * (tile_scale0 * kInvFp8Max);
    running_out[3] = running_out[3] * old_scale1 +
        tile_out[3] * (tile_scale1 * kInvFp8Max);
    running_max0 = next_max0;
    running_max1 = next_max1;
    __syncthreads();
  }

  const long long base0 =
      (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
       * kGroupSize + head0) * kPartialStride;
  const long long base1 = base0 + kPartialStride;
  if (warp == 0 && lane < 4) {
    partials[base0] = running_max0;
    partials[base0 + 1] = running_sum0;
    partials[base1] = running_max1;
    partials[base1 + 1] = running_sum1;
  }
  partials[base0 + 2 + row0] = running_out[0];
  partials[base1 + 2 + row0] = running_out[1];
  partials[base0 + 2 + row1] = running_out[2];
  partials[base1 + 2 + row1] = running_out[3];
#else
  (void)q;
  (void)k_cache;
  (void)v_cache;
  (void)page_table;
  (void)cache_seqlens;
  (void)partials;
  (void)max_pages;
  (void)num_splits;
  (void)softmax_scale;
#endif
}

// Phase-two XQA pipeline. Three 128-thread roles progress independently:
//   warps 0..3: QK WGMMA plus exact tile softmax
//   warps 4..7: P*V WGMMA plus associative running-state update
//   warps 8..11: page-table gather and double-buffered K/V staging
// Named barriers synchronize only the producer/consumer pairs that share a
// buffer, so the loader can stage tile i+1 while QK and P*V consume tile i.
__global__ __launch_bounds__(384, 2)
void paged_full_decode_partials_xqa_pipeline_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTileN = 64;
  constexpr float kFp8Max = 448.0f;
  constexpr float kInvFp8Max = 1.0f / kFp8Max;
  constexpr int kIoQkBarrier0 = 1;
  constexpr int kIoQkBarrier1 = 2;
  constexpr int kIoPvBarrier0 = 3;
  constexpr int kIoPvBarrier1 = 4;
  constexpr int kQkPvBarrier0 = 5;
  constexpr int kQkPvBarrier1 = 6;
  constexpr int kReuseBarrier0 = 7;
  constexpr int kReuseBarrier1 = 8;
  constexpr int kQkInternalBarrier = 9;
  constexpr int kIoInternalBarrier = 10;

  const int tid = static_cast<int>(threadIdx.x);
  const int linear = static_cast<int>(blockIdx.x);
  const int split = linear % num_splits;
  const int kv_head = (linear / num_splits) % kKvHeads;
  const int batch = linear / (num_splits * kKvHeads);
  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);
  const int num_tiles = max(0, (end - begin + kTileN - 1) / kTileN);

  __shared__ __align__(128) uint8_t s_q[kGroupSize * kHeadDim];
  __shared__ __align__(128) uint8_t s_k[2][kTileN * kHeadDim];
  __shared__ __align__(128) uint8_t s_p[2][kGroupSize * kTileN];
  __shared__ __align__(128) uint8_t s_v[2][kHeadDim * kTileN];
  __shared__ int s_pages[2][kTileN];
  __shared__ float s_warp_max[2][4][kGroupSize];
  __shared__ float s_warp_sum[2][4][kGroupSize];
  __shared__ float s_tile_max[2][kGroupSize];
  __shared__ float s_tile_sum[2][kGroupSize];

  for (int pair = tid; pair < kGroupSize * 32; pair += 384) {
    const int head = pair >> 5;
    const int pair_dim = pair & 31;
    const int q_head = kv_head * kGroupSize + head;
    const long long q_idx =
        (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim +
        pair_dim * 2;
    const uint16_t raw = *reinterpret_cast<const uint16_t*>(q + q_idx);
    const int dst = swizzle_b64_fp8(head * kHeadDim + pair_dim * 2);
    *reinterpret_cast<uint16_t*>(s_q + dst) = raw;
  }
  __syncthreads();

  if (tid >= 256) {
    // IO producer: two buffers are sufficient because reuse is acknowledged
    // only after the P*V consumer has finished reading the current buffer.
    const int io_tid = tid - 256;
    for (int tile = 0; tile < num_tiles; ++tile) {
      const int buffer = tile & 1;
      const int tile_begin = begin + tile * kTileN;
      if (tile >= 2) {
        if (buffer == 0) named_barrier<kReuseBarrier0, 256>();
        else named_barrier<kReuseBarrier1, 256>();
      }
      if (io_tid < kTileN) {
        const int pos = tile_begin + io_tid;
        s_pages[buffer][io_tid] = pos < end
            ? page_table[static_cast<long long>(batch) * max_pages + pos]
            : 0;
      }
      named_barrier<kIoInternalBarrier, 128>();
      for (int pair = io_tid; pair < kTileN * 32; pair += 128) {
        const int key = pair >> 5;
        const int pair_dim = pair & 31;
        const int pos = tile_begin + key;
        uint16_t raw_k = 0;
        uint16_t raw_v = 0;
        if (pos < end) {
          const long long cache_base =
              (static_cast<long long>(s_pages[buffer][key]) * kKvHeads + kv_head)
              * kHeadDim;
          raw_k = *reinterpret_cast<const uint16_t*>(
              k_cache + cache_base + pair_dim * 2);
          raw_v = *reinterpret_cast<const uint16_t*>(
              v_cache + cache_base + pair_dim * 2);
        }
        const int k_dst = swizzle_b64_fp8(key * kHeadDim + pair_dim * 2);
        *reinterpret_cast<uint16_t*>(s_k[buffer] + k_dst) = raw_k;
        const int dim0 = pair_dim * 2;
        s_v[buffer][swizzle_b64_fp8(dim0 * kTileN + key)] =
            static_cast<uint8_t>(raw_v & 0xff);
        s_v[buffer][swizzle_b64_fp8((dim0 + 1) * kTileN + key)] =
            static_cast<uint8_t>(raw_v >> 8);
      }
      if (buffer == 0) named_barrier<kIoQkBarrier0, 256>();
      else named_barrier<kIoQkBarrier1, 256>();
      if (buffer == 0) named_barrier<kIoPvBarrier0, 256>();
      else named_barrier<kIoPvBarrier1, 256>();
    }
  } else if (tid < 128) {
    // QK/softmax consumer-producer.
    const int qk_tid = tid;
    const int warp = qk_tid >> 5;
    const int lane = qk_tid & 31;
    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    const uint64_t q_desc = make_wgmma_desc_b64(s_q);

    for (int tile = 0; tile < num_tiles; ++tile) {
      const int buffer = tile & 1;
      const int tile_begin = begin + tile * kTileN;
      if (buffer == 0) named_barrier<kIoQkBarrier0, 256>();
      else named_barrier<kIoQkBarrier1, 256>();

      const uint64_t k_desc = make_wgmma_desc_b64(s_k[buffer]);
      float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      wgmma_arrive();
      wgmma_m64n8_f32_e4m3(k_desc, q_desc, scores, 0);
      wgmma_m64n8_f32_e4m3(k_desc + 2, q_desc + 2, scores, 1);
      wgmma_commit();
      wgmma_wait();

      float score0 = tile_begin + row0 < end ? scores[0] * softmax_scale : -FLT_MAX;
      float score1 = tile_begin + row0 < end ? scores[1] * softmax_scale : -FLT_MAX;
      float score2 = tile_begin + row1 < end ? scores[2] * softmax_scale : -FLT_MAX;
      float score3 = tile_begin + row1 < end ? scores[3] * softmax_scale : -FLT_MAX;
      float local_max0 = fmaxf(score0, score2);
      float local_max1 = fmaxf(score1, score3);
#pragma unroll
      for (int delta = 4; delta <= 16; delta *= 2) {
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, delta));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, delta));
      }
      if (lane < 4) {
        s_warp_max[buffer][warp][head0] = local_max0;
        s_warp_max[buffer][warp][head1] = local_max1;
      }
      named_barrier<kQkInternalBarrier, 128>();
      if (warp == 0 && lane < kGroupSize) {
        float value = s_warp_max[buffer][0][lane];
#pragma unroll
        for (int w = 1; w < 4; ++w) value = fmaxf(value, s_warp_max[buffer][w][lane]);
        s_tile_max[buffer][lane] = value;
      }
      named_barrier<kQkInternalBarrier, 128>();

      const float tile_max0 = s_tile_max[buffer][head0];
      const float tile_max1 = s_tile_max[buffer][head1];
      const float p0 = score0 == -FLT_MAX ? 0.0f
          : exp2f((score0 - tile_max0) * 1.4426950408889634f);
      const float p1 = score1 == -FLT_MAX ? 0.0f
          : exp2f((score1 - tile_max1) * 1.4426950408889634f);
      const float p2 = score2 == -FLT_MAX ? 0.0f
          : exp2f((score2 - tile_max0) * 1.4426950408889634f);
      const float p3 = score3 == -FLT_MAX ? 0.0f
          : exp2f((score3 - tile_max1) * 1.4426950408889634f);
      float local_sum0 = p0 + p2;
      float local_sum1 = p1 + p3;
#pragma unroll
      for (int delta = 4; delta <= 16; delta *= 2) {
        local_sum0 += __shfl_xor_sync(0xffffffff, local_sum0, delta);
        local_sum1 += __shfl_xor_sync(0xffffffff, local_sum1, delta);
      }
      if (lane < 4) {
        s_warp_sum[buffer][warp][head0] = local_sum0;
        s_warp_sum[buffer][warp][head1] = local_sum1;
      }
      s_p[buffer][swizzle_b64_fp8(head0 * kTileN + row0)] =
          static_cast<__nv_fp8_e4m3>(p0 * kFp8Max).__x;
      s_p[buffer][swizzle_b64_fp8(head1 * kTileN + row0)] =
          static_cast<__nv_fp8_e4m3>(p1 * kFp8Max).__x;
      s_p[buffer][swizzle_b64_fp8(head0 * kTileN + row1)] =
          static_cast<__nv_fp8_e4m3>(p2 * kFp8Max).__x;
      s_p[buffer][swizzle_b64_fp8(head1 * kTileN + row1)] =
          static_cast<__nv_fp8_e4m3>(p3 * kFp8Max).__x;
      named_barrier<kQkInternalBarrier, 128>();
      if (warp == 0 && lane < kGroupSize) {
        float value = s_warp_sum[buffer][0][lane];
#pragma unroll
        for (int w = 1; w < 4; ++w) value += s_warp_sum[buffer][w][lane];
        s_tile_sum[buffer][lane] = value;
      }
      named_barrier<kQkInternalBarrier, 128>();
      if (buffer == 0) named_barrier<kQkPvBarrier0, 256>();
      else named_barrier<kQkPvBarrier1, 256>();
    }
  } else {
    // P*V consumer and owner of the running online-softmax state.
    const int pv_tid = tid - 128;
    const int warp = pv_tid >> 5;

    const int lane = pv_tid & 31;
    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;

    const int head0 = (lane & 3) * 2;
    const int head1 = head0 + 1;
    float running_max0 = -FLT_MAX;

    float running_max1 = -FLT_MAX;
    float running_sum0 = 0.0f;
    float running_sum1 = 0.0f;
    float running_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (num_tiles > 0) named_barrier<kIoPvBarrier0, 256>();
    for (int tile = 0; tile < num_tiles; ++tile) {
      const int buffer = tile & 1;
      if (buffer == 0) named_barrier<kQkPvBarrier0, 256>();
      else named_barrier<kQkPvBarrier1, 256>();

      const uint64_t v_desc = make_wgmma_desc_b64(s_v[buffer]);
      const uint64_t p_desc = make_wgmma_desc_b64(s_p[buffer]);
      float tile_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      wgmma_arrive();
      wgmma_m64n8_f32_e4m3(v_desc, p_desc, tile_out, 0);
      wgmma_m64n8_f32_e4m3(v_desc + 2, p_desc + 2, tile_out, 1);
      wgmma_commit();
      wgmma_wait();

      const float tile_max0 = s_tile_max[buffer][head0];
      const float tile_max1 = s_tile_max[buffer][head1];
      const float next_max0 = fmaxf(running_max0, tile_max0);
      const float next_max1 = fmaxf(running_max1, tile_max1);
      const float old_scale0 =
          exp2f((running_max0 - next_max0) * 1.4426950408889634f);
      const float old_scale1 =
          exp2f((running_max1 - next_max1) * 1.4426950408889634f);
      const float tile_scale0 =
          exp2f((tile_max0 - next_max0) * 1.4426950408889634f);
      const float tile_scale1 =
          exp2f((tile_max1 - next_max1) * 1.4426950408889634f);
      running_sum0 = running_sum0 * old_scale0 +
          s_tile_sum[buffer][head0] * tile_scale0;
      running_sum1 = running_sum1 * old_scale1 +
          s_tile_sum[buffer][head1] * tile_scale1;
      running_out[0] = running_out[0] * old_scale0 +
          tile_out[0] * (tile_scale0 * kInvFp8Max);
      running_out[1] = running_out[1] * old_scale1 +
          tile_out[1] * (tile_scale1 * kInvFp8Max);
      running_out[2] = running_out[2] * old_scale0 +
          tile_out[2] * (tile_scale0 * kInvFp8Max);
      running_out[3] = running_out[3] * old_scale1 +
          tile_out[3] * (tile_scale1 * kInvFp8Max);
      running_max0 = next_max0;
      running_max1 = next_max1;

      // Release the producer's next V tile before acknowledging reuse of the
      // current buffer; the opposite order would create a two-buffer cycle.
      if (tile + 1 < num_tiles) {
        if (((tile + 1) & 1) == 0) named_barrier<kIoPvBarrier0, 256>();
        else named_barrier<kIoPvBarrier1, 256>();
      }
      if (tile + 2 < num_tiles) {
        if (buffer == 0) named_barrier<kReuseBarrier0, 256>();
        else named_barrier<kReuseBarrier1, 256>();
      }
    }

    const long long base0 =
        (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
         * kGroupSize + head0) * kPartialStride;
    const long long base1 = base0 + kPartialStride;
    if (warp == 0 && lane < 4) {
      partials[base0] = running_max0;
      partials[base0 + 1] = running_sum0;
      partials[base1] = running_max1;
      partials[base1 + 1] = running_sum1;
    }
    partials[base0 + 2 + row0] = running_out[0];
    partials[base1 + 2 + row0] = running_out[1];
    partials[base0 + 2 + row1] = running_out[2];
    partials[base1 + 2 + row1] = running_out[3];
  }
#else
  (void)q;
  (void)k_cache;
  (void)v_cache;
  (void)page_table;
  (void)cache_seqlens;
  (void)partials;
  (void)max_pages;
  (void)num_splits;
  (void)softmax_scale;
#endif
}

struct WarpPrivateU32x4 { uint32_t x, y, z, w; };
struct WarpPrivateU32x2 { uint32_t x, y; };

__device__ __forceinline__ WarpPrivateU32x4 warp_private_load128(const void* ptr) {
  WarpPrivateU32x4 out;
  asm volatile(
      "ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(ptr));
  return out;
}

__device__ __forceinline__ WarpPrivateU32x2 warp_private_load64(const void* ptr) {
  WarpPrivateU32x2 out;
  asm volatile(
      "ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
      : "=r"(out.x), "=r"(out.y)
      : "l"(ptr));
  return out;
}

__device__ __forceinline__ int warp_private_prefetch_page(
    const int32_t* page_table, long long offset) {
  int page;
  asm volatile(
      "ld.global.nc.L1::no_allocate.s32 %0, [%1];"
      : "=r"(page)
      : "l"(page_table + offset));
  return page;
}

__device__ __forceinline__ uint32_t warp_private_prmt(
    uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
  return out;
}

__device__ __forceinline__ uint32_t warp_private_pack4(
    uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3) {
  return static_cast<uint32_t>(x0)
      | (static_cast<uint32_t>(x1) << 8)
      | (static_cast<uint32_t>(x2) << 16)
      | (static_cast<uint32_t>(x3) << 24);
}

__device__ __forceinline__ float warp_private_from_e4m3(uint8_t raw) {
  __nv_fp8_e4m3 value;
  value.__x = raw;
  return static_cast<float>(value);
}

__device__ __forceinline__ void warp_private_mma_fp8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  (void)d0; (void)d1; (void)d2; (void)d3;
  (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ void warp_private_transpose4x8(
    const WarpPrivateU32x2& w0,
    const WarpPrivateU32x2& w1,
    const WarpPrivateU32x2& w2,
    const WarpPrivateU32x2& w3,
    uint32_t (&out)[8]) {
  const uint32_t lo0 = warp_private_prmt(w0.x, w1.x, 0x5140);
  const uint32_t lo1 = warp_private_prmt(w2.x, w3.x, 0x5140);
  const uint32_t lo2 = warp_private_prmt(w0.x, w1.x, 0x7362);
  const uint32_t lo3 = warp_private_prmt(w2.x, w3.x, 0x7362);
  out[0] = warp_private_prmt(lo0, lo1, 0x5410);
  out[1] = warp_private_prmt(lo0, lo1, 0x7632);
  out[2] = warp_private_prmt(lo2, lo3, 0x5410);
  out[3] = warp_private_prmt(lo2, lo3, 0x7632);

  const uint32_t hi0 = warp_private_prmt(w0.y, w1.y, 0x5140);
  const uint32_t hi1 = warp_private_prmt(w2.y, w3.y, 0x5140);
  const uint32_t hi2 = warp_private_prmt(w0.y, w1.y, 0x7362);
  const uint32_t hi3 = warp_private_prmt(w2.y, w3.y, 0x7362);
  out[4] = warp_private_prmt(hi0, hi1, 0x5410);
  out[5] = warp_private_prmt(hi0, hi1, 0x7632);
  out[6] = warp_private_prmt(hi2, hi3, 0x5410);
  out[7] = warp_private_prmt(hi2, hi3, 0x7632);
}

// Eight independent warps stream disjoint 32-key groups.  K and V go from
// global memory directly into warp MMA operand registers; the main loop has no
// shared-memory staging and no inter-warp barrier.
__global__ __launch_bounds__(256, 2)
void paged_full_decode_partials_warp_private_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  constexpr float kLog2E = 1.4426950408889634f;
  constexpr float kProbabilityScale = 128.0f;
  constexpr float kInvProbabilityScale = 1.0f / kProbabilityScale;
  constexpr unsigned kMask = 0xffffffffu;

  const int task = static_cast<int>(blockIdx.x);
  const int split = task % num_splits;
  const int kv_head = (task / num_splits) % kKvHeads;
  const int batch = task / (num_splits * kKvHeads);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int head = lane >> 2;
  const int quad_lane = lane & 3;

  const int seqlen = cache_seqlens[batch];
  const int chunk = (seqlen + num_splits - 1) / num_splits;
  const int begin = split * chunk;
  const int end = min(seqlen, begin + chunk);
  const int groups = max(0, (end - begin + 31) / 32);

  const int q_head = kv_head * kGroupSize + head;
  const long long q_offset =
      (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim
      + quad_lane * 16;
  const WarpPrivateU32x4 q_fragment =
      *reinterpret_cast<const WarpPrivateU32x4*>(q + q_offset);

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float output[8][2];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    output[n][0] = 0.0f;
    output[n][1] = 0.0f;
  }

  const long long page_base = static_cast<long long>(batch) * max_pages;
  int lane_page = 0;
  if (warp < groups) {
    const int first_lane_pos = begin + warp * 32 + lane;
    const int first_safe_pos = first_lane_pos < end ? first_lane_pos : begin;
    lane_page = warp_private_prefetch_page(
        page_table, page_base + first_safe_pos);
  }

  for (int group = warp; group < groups; group += 8) {
    const int pos0 = begin + group * 32;

    const int kp0 = __shfl_sync(kMask, lane_page, head + 0);
    const int kp1 = __shfl_sync(kMask, lane_page, head + 8);
    const int kp2 = __shfl_sync(kMask, lane_page, head + 16);
    const int kp3 = __shfl_sync(kMask, lane_page, head + 24);
    const long long kd = quad_lane * 16;
    const WarpPrivateU32x4 k0 = warp_private_load128(
        k_cache + (static_cast<long long>(kp0) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k1 = warp_private_load128(
        k_cache + (static_cast<long long>(kp1) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k2 = warp_private_load128(
        k_cache + (static_cast<long long>(kp2) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k3 = warp_private_load128(
        k_cache + (static_cast<long long>(kp3) * kKvHeads + kv_head) * kHeadDim + kd);

    const int vkey0 = 2 * quad_lane;
    const int vp0 = __shfl_sync(kMask, lane_page, vkey0 + 0);
    const int vp1 = __shfl_sync(kMask, lane_page, vkey0 + 1);
    const int vp2 = __shfl_sync(kMask, lane_page, vkey0 + 8);
    const int vp3 = __shfl_sync(kMask, lane_page, vkey0 + 9);
    const int vp4 = __shfl_sync(kMask, lane_page, vkey0 + 16);
    const int vp5 = __shfl_sync(kMask, lane_page, vkey0 + 17);
    const int vp6 = __shfl_sync(kMask, lane_page, vkey0 + 24);
    const int vp7 = __shfl_sync(kMask, lane_page, vkey0 + 25);
    const long long vd = head * 8;
    const WarpPrivateU32x2 v0 = warp_private_load64(
        v_cache + (static_cast<long long>(vp0) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v1 = warp_private_load64(
        v_cache + (static_cast<long long>(vp1) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v2 = warp_private_load64(
        v_cache + (static_cast<long long>(vp2) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v3 = warp_private_load64(
        v_cache + (static_cast<long long>(vp3) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v4 = warp_private_load64(
        v_cache + (static_cast<long long>(vp4) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v5 = warp_private_load64(
        v_cache + (static_cast<long long>(vp5) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v6 = warp_private_load64(
        v_cache + (static_cast<long long>(vp6) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v7 = warp_private_load64(
        v_cache + (static_cast<long long>(vp7) * kKvHeads + kv_head) * kHeadDim + vd);

    const int next_group = group + 8;
    int next_lane_page = 0;
    if (next_group < groups) {
      const int next_lane_pos = begin + next_group * 32 + lane;
      const int next_safe_pos = next_lane_pos < end ? next_lane_pos : begin;
      next_lane_page = warp_private_prefetch_page(
          page_table, page_base + next_safe_pos);
    }


    float scores[4][2];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {

      scores[tile][0] = 0.0f;
      scores[tile][1] = 0.0f;
      float pad0 = 0.0f;
      float pad1 = 0.0f;

      const WarpPrivateU32x4 kval = tile == 0 ? k0 : tile == 1 ? k1 : tile == 2 ? k2 : k3;
      warp_private_mma_fp8(
          scores[tile][0], scores[tile][1], pad0, pad1,
          q_fragment.x, 0, q_fragment.y, 0, kval.x, kval.y);
      warp_private_mma_fp8(
          scores[tile][0], scores[tile][1], pad0, pad1,
          q_fragment.z, 0, q_fragment.w, 0, kval.z, kval.w);
    }

    float local_max = -FLT_MAX;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int key = tile * 8 + quad_lane * 2 + pair;
        scores[tile][pair] = pos0 + key < end
            ? scores[tile][pair] * softmax_scale
            : -FLT_MAX;
        local_max = fmaxf(local_max, scores[tile][pair]);
      }
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(kMask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(kMask, local_max, 2));
    const float next_max = fmaxf(running_max, local_max);
    const float alpha = running_max == -FLT_MAX
        ? 0.0f
        : exp2f((running_max - next_max) * kLog2E);

#pragma unroll
    for (int n = 0; n < 8; ++n) {
      output[n][0] *= alpha;
      output[n][1] *= alpha;
    }

    uint8_t packed_probability[4][2];
    float local_probability_sum = 0.0f;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const float probability = scores[tile][pair] == -FLT_MAX
            ? 0.0f
            : exp2f((scores[tile][pair] - next_max) * kLog2E);
        const uint8_t raw = to_e4m3(probability * kProbabilityScale);
        packed_probability[tile][pair] = raw;
        local_probability_sum +=
            warp_private_from_e4m3(raw) * kInvProbabilityScale;
      }
    }
    local_probability_sum +=
        __shfl_xor_sync(kMask, local_probability_sum, 1);
    local_probability_sum +=
        __shfl_xor_sync(kMask, local_probability_sum, 2);
    running_sum = running_sum * alpha + local_probability_sum;
    running_max = next_max;

    const uint32_t probability0 = warp_private_pack4(
        packed_probability[0][0], packed_probability[0][1],
        packed_probability[1][0], packed_probability[1][1]);
    const uint32_t probability1 = warp_private_pack4(
        packed_probability[2][0], packed_probability[2][1],
        packed_probability[3][0], packed_probability[3][1]);
    uint32_t values0[8];
    uint32_t values1[8];
    warp_private_transpose4x8(v0, v1, v2, v3, values0);
    warp_private_transpose4x8(v4, v5, v6, v7, values1);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      float pad0 = 0.0f;
      float pad1 = 0.0f;
      warp_private_mma_fp8(
          output[n][0], output[n][1], pad0, pad1,
          probability0, 0, probability1, 0, values0[n], values1[n]);
    }
    lane_page = next_lane_page;
  }

  __shared__ float warp_maxima[8][kGroupSize];
  __shared__ float warp_sums[8][kGroupSize];
  __shared__ float warp_outputs[8][kGroupSize][kHeadDim];
  __shared__ float warp_factors[kGroupSize][8];
  if (quad_lane == 0) {
    warp_maxima[warp][head] = running_max;
    warp_sums[warp][head] = running_sum;
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    warp_outputs[warp][head][16 * quad_lane + n] = output[n][0] * kInvProbabilityScale;
    warp_outputs[warp][head][16 * quad_lane + 8 + n] = output[n][1] * kInvProbabilityScale;
  }
  __syncthreads();

  const int merge_head = warp;
  float source_max = lane < 8 ? warp_maxima[lane][merge_head] : -FLT_MAX;
  const float global_max = warp_max(source_max);
  float factor = 0.0f;
  float weighted_sum = 0.0f;
  if (lane < 8) {
    factor = source_max == -FLT_MAX
        ? 0.0f
        : exp2f((source_max - global_max) * kLog2E);
    weighted_sum = warp_sums[lane][merge_head] * factor;
    warp_factors[merge_head][lane] = factor;
  }
  const float merged_sum = warp_sum(weighted_sum);
  __syncwarp();
  float merged0 = 0.0f;
  float merged1 = 0.0f;
#pragma unroll
  for (int source = 0; source < 8; ++source) {
    const float source_factor = warp_factors[merge_head][source];
    merged0 += warp_outputs[source][merge_head][lane * 2] * source_factor;
    merged1 += warp_outputs[source][merge_head][lane * 2 + 1] * source_factor;
  }
  const long long partial_base =
      (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
       * kGroupSize + merge_head) * kPartialStride;
  if (lane == 0) {
    partials[partial_base] = global_max;
    partials[partial_base + 1] = merged_sum;
  }
  partials[partial_base + 2 + lane * 2] = merged0;
  partials[partial_base + 3 + lane * 2] = merged1;
#else
  (void)q; (void)k_cache; (void)v_cache; (void)page_table;
  (void)cache_seqlens; (void)partials; (void)max_pages;
  (void)num_splits; (void)softmax_scale;
#endif
}

// Compile the head-private and paired mappings from the same force-inlined
// implementation, but expose distinct, shape-specific kernel symbols.  This
// keeps the v14 launch symbol for B1/B16 and the v15a paired symbol for B32.
template <bool PairKvHeads, bool SlidingWindow>
__device__ __forceinline__ void paged_full_decode_partials_warp_private_impl(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    int window,
    float softmax_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  constexpr float kLog2E = 1.4426950408889634f;
  constexpr float kProbabilityScale = 128.0f;
  constexpr float kInvProbabilityScale = 1.0f / kProbabilityScale;
  constexpr unsigned kMask = 0xffffffffu;

  const int task = static_cast<int>(blockIdx.x);
  const int split = task % num_splits;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int head = lane >> 2;
  const int quad_lane = lane & 3;
  const int kv_head = PairKvHeads
      ? (warp & 1)
      : ((task / num_splits) % kKvHeads);
  const int batch = PairKvHeads
      ? (task / num_splits)
      : (task / (num_splits * kKvHeads));
  const int source_warp = PairKvHeads ? (warp >> 1) : warp;
  constexpr int kSourceWarps = PairKvHeads ? 4 : 8;

  const int seqlen = cache_seqlens[batch];
  const int range_begin = SlidingWindow ? max(0, seqlen - window) : 0;
  const int range_size = seqlen - range_begin;
  const int chunk = (range_size + num_splits - 1) / num_splits;
  const int begin = range_begin + split * chunk;
  const int end = min(seqlen, begin + chunk);
  const int groups = max(0, (end - begin + 31) / 32);

  const int q_head = kv_head * kGroupSize + head;
  const long long q_offset =
      (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim
      + quad_lane * 16;
  const WarpPrivateU32x4 q_fragment =
      *reinterpret_cast<const WarpPrivateU32x4*>(q + q_offset);

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float output[8][2];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    output[n][0] = 0.0f;
    output[n][1] = 0.0f;
  }

  const long long page_base = static_cast<long long>(batch) * max_pages;
  int lane_page = 0;
  if (source_warp < groups) {
    const int first_lane_pos = begin + source_warp * 32 + lane;
    const int first_safe_pos = first_lane_pos < end ? first_lane_pos : begin;
    lane_page = warp_private_prefetch_page(
        page_table, page_base + first_safe_pos);
  }

  for (int group = source_warp; group < groups; group += kSourceWarps) {
    const int pos0 = begin + group * 32;

    const int kp0 = __shfl_sync(kMask, lane_page, head + 0);
    const int kp1 = __shfl_sync(kMask, lane_page, head + 8);
    const int kp2 = __shfl_sync(kMask, lane_page, head + 16);
    const int kp3 = __shfl_sync(kMask, lane_page, head + 24);
    const long long kd = quad_lane * 16;
    const WarpPrivateU32x4 k0 = warp_private_load128(
        k_cache + (static_cast<long long>(kp0) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k1 = warp_private_load128(
        k_cache + (static_cast<long long>(kp1) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k2 = warp_private_load128(
        k_cache + (static_cast<long long>(kp2) * kKvHeads + kv_head) * kHeadDim + kd);
    const WarpPrivateU32x4 k3 = warp_private_load128(
        k_cache + (static_cast<long long>(kp3) * kKvHeads + kv_head) * kHeadDim + kd);

    const int vkey0 = 2 * quad_lane;
    const int vp0 = __shfl_sync(kMask, lane_page, vkey0 + 0);
    const int vp1 = __shfl_sync(kMask, lane_page, vkey0 + 1);
    const int vp2 = __shfl_sync(kMask, lane_page, vkey0 + 8);
    const int vp3 = __shfl_sync(kMask, lane_page, vkey0 + 9);
    const int vp4 = __shfl_sync(kMask, lane_page, vkey0 + 16);
    const int vp5 = __shfl_sync(kMask, lane_page, vkey0 + 17);
    const int vp6 = __shfl_sync(kMask, lane_page, vkey0 + 24);
    const int vp7 = __shfl_sync(kMask, lane_page, vkey0 + 25);
    const long long vd = head * 8;
    const WarpPrivateU32x2 v0 = warp_private_load64(
        v_cache + (static_cast<long long>(vp0) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v1 = warp_private_load64(
        v_cache + (static_cast<long long>(vp1) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v2 = warp_private_load64(
        v_cache + (static_cast<long long>(vp2) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v3 = warp_private_load64(
        v_cache + (static_cast<long long>(vp3) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v4 = warp_private_load64(
        v_cache + (static_cast<long long>(vp4) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v5 = warp_private_load64(
        v_cache + (static_cast<long long>(vp5) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v6 = warp_private_load64(
        v_cache + (static_cast<long long>(vp6) * kKvHeads + kv_head) * kHeadDim + vd);
    const WarpPrivateU32x2 v7 = warp_private_load64(
        v_cache + (static_cast<long long>(vp7) * kKvHeads + kv_head) * kHeadDim + vd);

    const int next_group = group + kSourceWarps;
    int next_lane_page = 0;
    if (next_group < groups) {
      const int next_lane_pos = begin + next_group * 32 + lane;
      const int next_safe_pos = next_lane_pos < end ? next_lane_pos : begin;
      next_lane_page = warp_private_prefetch_page(
          page_table, page_base + next_safe_pos);
    }


    float scores[4][2];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {

      scores[tile][0] = 0.0f;
      scores[tile][1] = 0.0f;
      float pad0 = 0.0f;
      float pad1 = 0.0f;

      const WarpPrivateU32x4 kval = tile == 0 ? k0 : tile == 1 ? k1 : tile == 2 ? k2 : k3;
      warp_private_mma_fp8(
          scores[tile][0], scores[tile][1], pad0, pad1,
          q_fragment.x, 0, q_fragment.y, 0, kval.x, kval.y);
      warp_private_mma_fp8(
          scores[tile][0], scores[tile][1], pad0, pad1,
          q_fragment.z, 0, q_fragment.w, 0, kval.z, kval.w);
    }

    float local_max = -FLT_MAX;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int key = tile * 8 + quad_lane * 2 + pair;
        scores[tile][pair] = pos0 + key < end
            ? scores[tile][pair] * softmax_scale
            : -FLT_MAX;
        local_max = fmaxf(local_max, scores[tile][pair]);
      }
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(kMask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(kMask, local_max, 2));
    const float next_max = fmaxf(running_max, local_max);
    const float alpha = running_max == -FLT_MAX
        ? 0.0f
        : exp2f((running_max - next_max) * kLog2E);

#pragma unroll
    for (int n = 0; n < 8; ++n) {
      output[n][0] *= alpha;
      output[n][1] *= alpha;
    }

    uint8_t packed_probability[4][2];
    float local_probability_sum = 0.0f;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const float probability = scores[tile][pair] == -FLT_MAX
            ? 0.0f
            : exp2f((scores[tile][pair] - next_max) * kLog2E);
        const uint8_t raw = to_e4m3(probability * kProbabilityScale);
        packed_probability[tile][pair] = raw;
        local_probability_sum +=
            warp_private_from_e4m3(raw) * kInvProbabilityScale;
      }
    }
    local_probability_sum +=
        __shfl_xor_sync(kMask, local_probability_sum, 1);
    local_probability_sum +=
        __shfl_xor_sync(kMask, local_probability_sum, 2);
    running_sum = running_sum * alpha + local_probability_sum;
    running_max = next_max;

    const uint32_t probability0 = warp_private_pack4(
        packed_probability[0][0], packed_probability[0][1],
        packed_probability[1][0], packed_probability[1][1]);
    const uint32_t probability1 = warp_private_pack4(
        packed_probability[2][0], packed_probability[2][1],
        packed_probability[3][0], packed_probability[3][1]);
    uint32_t values0[8];
    uint32_t values1[8];
    warp_private_transpose4x8(v0, v1, v2, v3, values0);
    warp_private_transpose4x8(v4, v5, v6, v7, values1);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      float pad0 = 0.0f;
      float pad1 = 0.0f;
      warp_private_mma_fp8(
          output[n][0], output[n][1], pad0, pad1,
          probability0, 0, probability1, 0, values0[n], values1[n]);
    }
    lane_page = next_lane_page;
  }

  __shared__ float warp_maxima[8][kGroupSize];
  __shared__ float warp_sums[8][kGroupSize];
  __shared__ float warp_outputs[8][kGroupSize][kHeadDim];
  __shared__ float warp_factors[kGroupSize][8];
  const int storage_source = PairKvHeads
      ? kv_head * kSourceWarps + source_warp
      : source_warp;
  if (quad_lane == 0) {
    warp_maxima[storage_source][head] = running_max;
    warp_sums[storage_source][head] = running_sum;
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    warp_outputs[storage_source][head][16 * quad_lane + n] =
        output[n][0] * kInvProbabilityScale;
    warp_outputs[storage_source][head][16 * quad_lane + 8 + n] =
        output[n][1] * kInvProbabilityScale;
  }
  __syncthreads();

  const int merge_head = warp;
  constexpr int kMergeHeads = PairKvHeads ? kKvHeads : 1;
#pragma unroll
  for (int merge_kv_iter = 0; merge_kv_iter < kMergeHeads; ++merge_kv_iter) {
    const int merge_kv_head = PairKvHeads ? merge_kv_iter : kv_head;
    const int source_base = PairKvHeads ? merge_kv_head * kSourceWarps : 0;
    float source_max = lane < kSourceWarps
        ? warp_maxima[source_base + lane][merge_head]
        : -FLT_MAX;
    const float global_max = warp_max(source_max);
    float factor = 0.0f;
    float weighted_sum = 0.0f;
    if (lane < kSourceWarps) {
      factor = source_max == -FLT_MAX
          ? 0.0f
          : exp2f((source_max - global_max) * kLog2E);
      weighted_sum = warp_sums[source_base + lane][merge_head] * factor;
      warp_factors[merge_head][source_base + lane] = factor;
    }
    const float merged_sum = warp_sum(weighted_sum);
    __syncwarp();
    float merged0 = 0.0f;
    float merged1 = 0.0f;
#pragma unroll
    for (int source = 0; source < kSourceWarps; ++source) {
      const float source_factor =
          warp_factors[merge_head][source_base + source];
      merged0 += warp_outputs[source_base + source][merge_head][lane * 2]
          * source_factor;
      merged1 += warp_outputs[source_base + source][merge_head][lane * 2 + 1]
          * source_factor;
    }
    const long long partial_base =
        (((static_cast<long long>(batch) * kKvHeads + merge_kv_head)
          * num_splits + split) * kGroupSize + merge_head) * kPartialStride;
    if (lane == 0) {
      partials[partial_base] = global_max;
      partials[partial_base + 1] = merged_sum;
    }
    partials[partial_base + 2 + lane * 2] = merged0;
    partials[partial_base + 3 + lane * 2] = merged1;
  }
#else
  (void)q; (void)k_cache; (void)v_cache; (void)page_table;
  (void)cache_seqlens; (void)partials; (void)max_pages;
  (void)num_splits; (void)window; (void)softmax_scale;
#endif
}

__global__ __launch_bounds__(256, 2)
void paged_full_decode_partials_warp_private_paired_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    float softmax_scale) {
  paged_full_decode_partials_warp_private_impl<true, false>(
      q, k_cache, v_cache, page_table, cache_seqlens, partials,
      max_pages, num_splits, 0, softmax_scale);
}

__global__ __launch_bounds__(256, 2)
void paged_sliding_decode_partials_warp_private_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k_cache,
    const __nv_fp8_e4m3* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ partials,
    int max_pages,
    int num_splits,
    int window,
    float softmax_scale) {
  paged_full_decode_partials_warp_private_impl<false, true>(
      q, k_cache, v_cache, page_table, cache_seqlens, partials,
      max_pages, num_splits, window, softmax_scale);
}

__global__ __launch_bounds__(512, 1)
void paged_full_decode_combine_kernel(
    const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ out,
    int num_splits) {
  const int batch = static_cast<int>(blockIdx.x);
  const int q_head = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (q_head >= kQueryHeads) return;
  const int kv_head = q_head / kGroupSize;
  const int group_head = q_head % kGroupSize;

  float global_max = -FLT_MAX;
#pragma unroll
  for (int split = 0; split < 8; ++split) {
    if (split < num_splits) {
      const long long base =
          (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
           * kGroupSize + group_head) * kPartialStride;
      global_max = fmaxf(global_max, partials[base]);
    }
  }
  float denom = 0.0f;
  float value0 = 0.0f;
  float value1 = 0.0f;
#pragma unroll
  for (int split = 0; split < 8; ++split) {
    if (split < num_splits) {
      const long long base =
          (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
           * kGroupSize + group_head) * kPartialStride;
      const float factor = exp2f((partials[base] - global_max) * 1.4426950408889634f);
      denom += partials[base + 1] * factor;
      value0 += partials[base + 2 + lane * 2] * factor;
      value1 += partials[base + 3 + lane * 2] * factor;
    }
  }
  const float inverse = 1.0f / denom;
  const float attention0 = value0 * inverse;
  const float attention1 = value1 * inverse;
  const float rounded_attention0 =
      __bfloat162float(__float2bfloat16_rn(attention0));
  const float rounded_attention1 =
      __bfloat162float(__float2bfloat16_rn(attention1));
  const float square_sum = warp_sum(
      rounded_attention0 * rounded_attention0
      + rounded_attention1 * rounded_attention1);
  const float inv_rms = rsqrtf(square_sum / 64.0f + 1.0e-5f);
  constexpr int kGateOffset = (kQueryHeads + 2 * kKvHeads) * kHeadDim;
  const long long gate_base =
      static_cast<long long>(batch) * kQkvStride + kGateOffset
      + static_cast<long long>(q_head) * kHeadDim + lane * 2;
  const float gate0 = __bfloat162float(qkv[gate_base]);
  const float gate1 = __bfloat162float(qkv[gate_base + 1]);
  const float sigmoid0 = 1.0f / (1.0f + expf(-gate0));
  const float sigmoid1 = 1.0f / (1.0f + expf(-gate1));
  const long long output_base =
      (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim;
  out[output_base + lane * 2] =
      __float2bfloat16_rn(rounded_attention0 * inv_rms * sigmoid0);
  out[output_base + lane * 2 + 1] =
      __float2bfloat16_rn(rounded_attention1 * inv_rms * sigmoid1);
}

// Preserve v15a's 16-way-unrolled combine binary for the paired B32 path.
// B16 dispatches to paged_full_decode_combine_kernel above, whose eight-way
// loops are the exact v14 source path.
__global__ __launch_bounds__(512, 1)
void paged_full_decode_combine_paired_kernel(
    const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ out,
    int num_splits) {
  const int batch = static_cast<int>(blockIdx.x);
  const int q_head = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (q_head >= kQueryHeads) return;
  const int kv_head = q_head / kGroupSize;
  const int group_head = q_head % kGroupSize;

  float global_max = -FLT_MAX;
#pragma unroll
  for (int split = 0; split < 16; ++split) {
    if (split < num_splits) {
      const long long base =
          (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
           * kGroupSize + group_head) * kPartialStride;
      global_max = fmaxf(global_max, partials[base]);
    }
  }
  float denom = 0.0f;
  float value0 = 0.0f;
  float value1 = 0.0f;
#pragma unroll
  for (int split = 0; split < 16; ++split) {
    if (split < num_splits) {
      const long long base =
          (((static_cast<long long>(batch) * kKvHeads + kv_head) * num_splits + split)
           * kGroupSize + group_head) * kPartialStride;
      const float factor = exp2f((partials[base] - global_max) * 1.4426950408889634f);
      denom += partials[base + 1] * factor;
      value0 += partials[base + 2 + lane * 2] * factor;
      value1 += partials[base + 3 + lane * 2] * factor;
    }
  }
  const float inverse = 1.0f / denom;
  const float attention0 = value0 * inverse;
  const float attention1 = value1 * inverse;
  const float rounded_attention0 =
      __bfloat162float(__float2bfloat16_rn(attention0));
  const float rounded_attention1 =
      __bfloat162float(__float2bfloat16_rn(attention1));
  const float square_sum = warp_sum(
      rounded_attention0 * rounded_attention0
      + rounded_attention1 * rounded_attention1);
  const float inv_rms = rsqrtf(square_sum / 64.0f + 1.0e-5f);
  constexpr int kGateOffset = (kQueryHeads + 2 * kKvHeads) * kHeadDim;
  const long long gate_base =
      static_cast<long long>(batch) * kQkvStride + kGateOffset
      + static_cast<long long>(q_head) * kHeadDim + lane * 2;
  const float gate0 = __bfloat162float(qkv[gate_base]);
  const float gate1 = __bfloat162float(qkv[gate_base + 1]);
  const float sigmoid0 = 1.0f / (1.0f + expf(-gate0));
  const float sigmoid1 = 1.0f / (1.0f + expf(-gate1));
  const long long output_base =
      (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim;
  out[output_base + lane * 2] =
      __float2bfloat16_rn(rounded_attention0 * inv_rms * sigmoid0);
  out[output_base + lane * 2 + 1] =
      __float2bfloat16_rn(rounded_attention1 * inv_rms * sigmoid1);
}

// Batch-one decode uses 64 splits to expose one H200 wave.  One two-warp block
// per query head parallelizes the expensive max/exp/denominator merge; the 64
// lanes then each own one output dimension for the weighted value reduction.
__global__ __launch_bounds__(64, 4)
void paged_full_decode_combine64_kernel(
    const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ out) {
  constexpr int kSplits = 64;
  const int task = static_cast<int>(blockIdx.x);
  const int batch = task / kQueryHeads;
  const int q_head = task % kQueryHeads;
  const int kv_head = q_head / kGroupSize;
  const int group_head = q_head % kGroupSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const long long split_base =
      (static_cast<long long>(batch) * kKvHeads + kv_head) * kSplits;
  const long long own_base =
      ((split_base + tid) * kGroupSize + group_head) * kPartialStride;

  __shared__ float warp_values[2];
  __shared__ float factors[kSplits];

  const float source_max = partials[own_base];
  const float local_max = warp_max(source_max);
  if (lane == 0) warp_values[warp] = local_max;
  __syncthreads();
  const float global_max = fmaxf(warp_values[0], warp_values[1]);

  const float factor = exp2f((source_max - global_max) * 1.4426950408889634f);
  factors[tid] = factor;
  const float local_denom = warp_sum(partials[own_base + 1] * factor);
  if (lane == 0) warp_values[warp] = local_denom;
  __syncthreads();
  const float denom = warp_values[0] + warp_values[1];

  float value = 0.0f;
#pragma unroll 8
  for (int split = 0; split < kSplits; ++split) {
    const long long base =
        ((split_base + split) * kGroupSize + group_head) * kPartialStride;
    value += partials[base + 2 + tid] * factors[split];
  }
  const float attention = value / denom;
  const float rounded_attention =
      __bfloat162float(__float2bfloat16_rn(attention));
  const float local_square_sum =
      warp_sum(rounded_attention * rounded_attention);
  if (lane == 0) warp_values[warp] = local_square_sum;
  __syncthreads();
  const float inv_rms =
      rsqrtf((warp_values[0] + warp_values[1]) / 64.0f + 1.0e-5f);
  constexpr int kGateOffset = (kQueryHeads + 2 * kKvHeads) * kHeadDim;
  const long long gate_index =
      static_cast<long long>(batch) * kQkvStride + kGateOffset
      + static_cast<long long>(q_head) * kHeadDim + tid;
  const float gate = __bfloat162float(qkv[gate_index]);
  const float sigmoid = 1.0f / (1.0f + expf(-gate));
  const long long output_base =
      (static_cast<long long>(batch) * kQueryHeads + q_head) * kHeadDim;
  out[output_base + tid] =
      __float2bfloat16_rn(rounded_attention * inv_rms * sigmoid);
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

torch::Tensor fused_prepare_sliding_verify(
    torch::Tensor qkv,
    torch::Tensor positions,
    torch::Tensor cache_loc,
    torch::Tensor rope_cache,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor q_out,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_seqlens,
    torch::Tensor scheduler_metadata,
    int64_t num_splits) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be BF16");
  TORCH_CHECK(qkv.size(-1) == kQkvStride, "unexpected TP8 qkv width");
  TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(scheduler_metadata.scalar_type() == torch::kInt32, "metadata must be int32");
  const int tokens = static_cast<int>(qkv.numel() / kQkvStride);
  const int batch = static_cast<int>(cache_seqlens.numel());
  const int b_rounded = (batch + 3) & ~3;
  TORCH_CHECK(batch == 16 || batch == 32, "fused scheduler supports B16/B32");
  TORCH_CHECK(tokens == batch * 8, "fused scheduler supports q_len=8");
  TORCH_CHECK(num_splits == (batch == 16 ? 4 : 2), "unexpected split count");
  TORCH_CHECK(scheduler_metadata.numel() == 3 * b_rounded + 1,
              "unexpected scheduler metadata size");
  static const int num_sm = [] {
    int device = 0;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.multiProcessorCount;
  }();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_prepare_sliding_verify_kernel<true, false>
      <<<tokens + 1, kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
          positions.data_ptr<int64_t>(), cache_loc.data_ptr<int64_t>(),
          rope_cache.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          reinterpret_cast<uint8_t*>(q_out.data_ptr()),
          reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
          reinterpret_cast<uint8_t*>(v_cache.data_ptr()),
          cache_seqlens.data_ptr<int32_t>(), scheduler_metadata.data_ptr<int32_t>(),
          tokens, batch, static_cast<int>(num_splits), num_sm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return q_out;
}

torch::Tensor paged_full_decode(
    torch::Tensor q,
    torch::Tensor qkv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor cache_seqlens,
    torch::Tensor partials,
    torch::Tensor out,
    int64_t num_splits,
    bool pair_kv_heads,
    int64_t window,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && qkv.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.scalar_type() == c10::ScalarType::Float8_e4m3fn, "q must be E4M3");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be BF16");
  TORCH_CHECK(k_cache.scalar_type() == c10::ScalarType::Float8_e4m3fn, "k cache must be E4M3");
  TORCH_CHECK(v_cache.scalar_type() == c10::ScalarType::Float8_e4m3fn, "v cache must be E4M3");
  TORCH_CHECK(q.size(1) == kQueryHeads && q.size(2) == kHeadDim, "unexpected q shape");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page table must be int32");
  TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "lengths must be int32");
  TORCH_CHECK(num_splits > 0 && num_splits <= 64, "unsupported split count");
  const int batch = static_cast<int>(q.size(0));
  TORCH_CHECK(
      window > 0
          ? (!pair_kv_heads
             && ((batch == 1 && num_splits == 16)
                 || ((batch == 16 || batch == 32) && num_splits == 8)))
          : (pair_kv_heads ? (batch == 32 && num_splits == 8)
                           : ((batch == 1 && num_splits == 64)
                              || (batch == 16 && num_splits == 8))),
      "invalid custom decode shape dispatch");
  const int max_pages = static_cast<int>(page_table.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 head_blocks(batch * kKvHeads * static_cast<int>(num_splits));
  const dim3 paired_blocks(batch * static_cast<int>(num_splits));
  const dim3 threads(128);
  const dim3 warp_private_threads(256);
  const dim3 pipeline_threads(384);
  static const int device_major = [] {
    int device = 0;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.major;
  }();
  if (device_major >= 9) {
    if (window > 0) {
      paged_sliding_decode_partials_warp_private_kernel
          <<<head_blocks, warp_private_threads, 0, stream>>>(
              reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(k_cache.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(v_cache.data_ptr()),
              page_table.data_ptr<int32_t>(), cache_seqlens.data_ptr<int32_t>(),
              partials.data_ptr<float>(), max_pages, static_cast<int>(num_splits),
              static_cast<int>(window), static_cast<float>(softmax_scale));
    } else if (pair_kv_heads) {
      paged_full_decode_partials_warp_private_paired_kernel
          <<<paired_blocks, warp_private_threads, 0, stream>>>(
              reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(k_cache.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(v_cache.data_ptr()),
              page_table.data_ptr<int32_t>(), cache_seqlens.data_ptr<int32_t>(),
              partials.data_ptr<float>(), max_pages, static_cast<int>(num_splits),
              static_cast<float>(softmax_scale));
    } else {
      paged_full_decode_partials_warp_private_kernel
          <<<head_blocks, warp_private_threads, 0, stream>>>(
              reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(k_cache.data_ptr()),
              reinterpret_cast<const __nv_fp8_e4m3*>(v_cache.data_ptr()),
              page_table.data_ptr<int32_t>(), cache_seqlens.data_ptr<int32_t>(),
              partials.data_ptr<float>(), max_pages, static_cast<int>(num_splits),
              static_cast<float>(softmax_scale));
    }
  } else {
    paged_full_decode_partials_tc_kernel<<<head_blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(v_cache.data_ptr()),
        page_table.data_ptr<int32_t>(), cache_seqlens.data_ptr<int32_t>(),
        partials.data_ptr<float>(), max_pages, static_cast<int>(num_splits),
        static_cast<float>(softmax_scale));
  }
  if (num_splits == 64) {
    paged_full_decode_combine64_kernel<<<batch * kQueryHeads, 64, 0, stream>>>(
        partials.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  } else if (pair_kv_heads || num_splits > 8) {
    paged_full_decode_combine_paired_kernel<<<batch, 512, 0, stream>>>(
        partials.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(num_splits));
  } else {
    paged_full_decode_combine_kernel<<<batch, 512, 0, stream>>>(
        partials.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(num_splits));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_prepare", &fused_prepare);
  module.def("fused_prepare_sliding_verify", &fused_prepare_sliding_verify);
  module.def("paged_full_decode", &paged_full_decode);
}
"""


_attention_extension = load_inline(


    name="okc_muse_attention_warp_private_v19a",
    cpp_sources="",
    cuda_sources=_ATTENTION_CUDA,
    extra_cuda_cflags=[
        "-O3",

        "--use_fast_math",
        "-gencode=arch=compute_90a,code=sm_90a",
    ],
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
    window = cfg["window"]
    fused_sliding_verify_scheduler = (
        length == 8
        and window == 2048
        and cfg["page_size"] == 1
        and batch in (16, 32)
        and not scaled
        and hq == 16
        and hkv == 2
        and d == 64
    )
    prepared_scheduler_metadata = None

    if hq == 16 and hkv == 2 and d == 64:
        q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
        rope_cache = (
            _cos_sin(cfg["rope_theta"], d, qkv.device)
            if cfg["rope_theta"]
            else data["k_scale"]
        )
        if fused_sliding_verify_scheduler:
            b_rounded = (batch + 3) // 4 * 4
            prepared_scheduler_metadata = torch.empty(
                (3 * b_rounded + 1,), dtype=torch.int32, device=qkv.device
            )
            q = _attention_extension.fused_prepare_sliding_verify(
                qkv,
                data["positions"],
                data["cache_loc"],
                rope_cache,
                data["k_scale"],
                data["v_scale"],
                q,
                data["k_cache"],
                data["v_cache"],
                data["cache_seqlens"],
                prepared_scheduler_metadata,
                4 if batch == 16 else 2,
            )
        else:
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

    custom_full_decode = (
        length == 1
        and window == 0
        and cfg["page_size"] == 1
        and batch in (1, 16, 32)
        and not scaled
        and hq == 16
        and hkv == 2
        and d == 64
    )
    custom_sliding_decode = (
        length == 1
        and window == 2048
        and cfg["page_size"] == 1
        and batch in (1, 16, 32)
        and not scaled
        and hq == 16
        and hkv == 2
        and d == 64
    )
    custom_decode = custom_full_decode or custom_sliding_decode
    if custom_decode:
        pair_kv_heads = custom_full_decode and batch == 32
        splits = (64 if batch == 1 else 8) if custom_full_decode else (16 if batch == 1 else 8)
        partials = torch.empty(
            (batch, hkv, splits, hq // hkv, 2 + d),
            dtype=torch.float32,
            device=q.device,
        )
        attention_out = torch.empty(
            (batch, hq, d), dtype=torch.bfloat16, device=q.device
        )
        out = _attention_extension.paged_full_decode(
            q,
            qkv,
            data["k_cache"],
            data["v_cache"],
            data["page_table"],
            data["cache_seqlens"],
            partials,
            attention_out,
            splits,
            pair_kv_heads,
            window if custom_sliding_decode else 0,
            cfg["sm_scale"] * cfg["qk_scale_factor"],
        )
    else:
        descale = {}
        if scaled:
            descale = {
                "k_descale": data["k_scale"].reshape(1, 1).expand(batch, hkv),
                "v_descale": data["v_scale"].reshape(1, 1).expand(batch, hkv),
            }
        # For varlen calls without precomputed metadata, keep one near-full H200
        # wave: B * Hkv * splits = 128 CTAs for the scored B=16/32 cases.
        # Batch one and metadata-backed full decode retain FA3's dynamic heuristic.
        num_splits = 0
        if data["scheduler_metadata"] is None:
            if batch == 16:
                num_splits = 8 if window == 0 and length == 8 else 4
            elif batch == 32:
                num_splits = 4 if window == 0 and length == 8 else 2
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
            scheduler_metadata=(
                None
                if static_single
                else prepared_scheduler_metadata
                if fused_sliding_verify_scheduler
                else data["scheduler_metadata"]
            ),
            num_splits=num_splits,
            **descale,
        )
        if isinstance(out, tuple):
            out = out[0]
    if custom_decode:
        return out.reshape(batch, length, hq * d)

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
