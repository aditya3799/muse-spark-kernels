"""Fused QKV m=1 Codec E GEMV prototype.

The geometry matches submission.py's GVCU route (WR=1, WK=8, KU=16,
RT=1).  Only the C12 high-code loader changes: four warp-interleaved Codec E
substreams recreate the same 16 code bytes before the unchanged dec1 + MMA
sequence.  This file is a local/H200 experiment, not production dispatch.
"""

from __future__ import annotations

import hashlib

import torch
from torch.utils.cpp_extension import load_inline

from codec_e import build_plan, pack_warp_interleaved, unpack_warp_interleaved


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 load_nc4(const void* pointer) {
  uint4 value;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer));
  return value;
}

__device__ __forceinline__ uint32_t load_nc1(const uint32_t* pointer) {
  uint32_t value;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];"
               : "=r"(value) : "l"(pointer));
  return value;
}

__device__ __forceinline__ uint32_t permute(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t result;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(selector));
  return result;
}

__device__ __forceinline__ void decode_c12(
    uint32_t low0, uint32_t low1, uint32_t codes, uint32_t base4, uint32_t* output) {
  const uint32_t high_low = (codes & 0x87878787u) + base4;
  const uint32_t high_high = (__funnelshift_r(codes, codes, 4) & 0x87878787u) + base4;
  output[0] = permute(low0, high_low, 0x5140);
  output[1] = permute(low0, high_low, 0x7362);
  output[2] = permute(low1, high_high, 0x5140);
  output[3] = permute(low1, high_high, 0x7362);
}

__device__ __forceinline__ void mma_bf16(
    float* accumulator, const uint32_t* weights, uint32_t x0, uint32_t x1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]),
        "+f"(accumulator[2]), "+f"(accumulator[3])
      : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]),
        "r"(x0), "r"(x1));
}

template <int ST>
__global__ void __launch_bounds__(256) codec_e_qkv_m1_kernel(
    const uint4* __restrict__ low,
    const uint32_t* __restrict__ stream_words,
    const int32_t* __restrict__ stream_offsets,
    const uint8_t* __restrict__ lut,
    const int32_t* __restrict__ suffix_bits,
    const int32_t* __restrict__ bases,
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ escapes,
    __nv_bfloat16* __restrict__ output,
    uint32_t base4,
    int tiles) {
  constexpr int KC = 128, WK = 8, KU = 16, SUB = 4;
  __shared__ float4 reduction[WK][32];
  __shared__ uint8_t shared_lut[256];
  __shared__ uint8_t shared_suffix_bits[8];
  __shared__ uint16_t shared_bases[8];
  for (int index = threadIdx.x; index < 256; index += blockDim.x)
    shared_lut[index] = lut[index];
  if (threadIdx.x < 8) {
    shared_suffix_bits[threadIdx.x] = (uint8_t)suffix_bits[threadIdx.x];
    shared_bases[threadIdx.x] = (uint16_t)bases[threadIdx.x];
  }
  __syncthreads();

  const int tile = blockIdx.x;
  if (tile >= tiles) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int group = lane >> 2, thread_k = lane & 3;
  const int chunk0 = warp * KU;
  const size_t low_base = ((size_t)tile * KC + chunk0) * 2 * 32 + lane;
  const __nv_bfloat16* x_base = x + chunk0 * 64 + thread_k * 16;

  uint4 low_ring[ST][2], x_ring[ST][2];
  auto prefetch = [&](int unit, int slot) {
    low_ring[slot][0] = load_nc4(low + low_base + (size_t)unit * 2 * 32);
    low_ring[slot][1] = load_nc4(low + low_base + (size_t)unit * 2 * 32 + 32);
    x_ring[slot][0] = load_nc4(x_base + unit * 64);
    x_ring[slot][1] = load_nc4(x_base + unit * 64 + 8);
  };
#pragma unroll
  for (int slot = 0; slot < ST; ++slot) prefetch(slot, slot);

  uint64_t buffers[SUB];
  int available[SUB], next_word[SUB];
#pragma unroll
  for (int substream = 0; substream < SUB; ++substream) {
    const int offset = stream_offsets[((tile * WK + warp) * SUB) + substream];
    buffers[substream] =
        (uint64_t)load_nc1(stream_words + offset + lane) |
        ((uint64_t)load_nc1(stream_words + offset + 32 + lane) << 32);
    available[substream] = 64;
    next_word[substream] = 2;
  }

  uint32_t escape = escapes[((tile * WK + warp) * 32) + lane];
  uint32_t next_escape = __shfl_sync(0xffffffffu, escape, 0);
  int escape_position = 0;
  float accumulators[2][4] = {};

#pragma unroll
  for (int unit = 0; unit < KU; ++unit) {
    const int slot = unit % ST;
    uint32_t code_words[4] = {};
#pragma unroll
    for (int position = 0; position < 4; ++position) {
#pragma unroll
      for (int substream = 0; substream < SUB; ++substream) {
        uint64_t buffer = buffers[substream];
        const uint32_t prefix = (uint32_t)buffer & 7u;
        const uint32_t width = shared_suffix_bits[prefix];
        const uint32_t suffix = ((uint32_t)(buffer >> 3)) & ((1u << width) - 1u);
        code_words[position] |=
            (uint32_t)shared_lut[shared_bases[prefix] + suffix] << (8 * substream);
        const int consumed = 3 + (int)width;
        buffer >>= consumed;
        available[substream] -= consumed;
        if (available[substream] < 11 && unit * 4 + position + 1 < 64) {
          const int offset = stream_offsets[((tile * WK + warp) * SUB) + substream];
          buffer |= (uint64_t)load_nc1(
              stream_words + offset + next_word[substream] * 32 + lane)
              << available[substream];
          available[substream] += 32;
          ++next_word[substream];
        }
        buffers[substream] = buffer;
      }
    }

    uint32_t weights[16];
    const uint4 low0 = low_ring[slot][0], low1 = low_ring[slot][1];
    const uint4 x0 = x_ring[slot][0], x1 = x_ring[slot][1];
    decode_c12(low0.x, low0.y, code_words[0], base4, weights + 0);
    decode_c12(low0.z, low0.w, code_words[1], base4, weights + 4);
    decode_c12(low1.x, low1.y, code_words[2], base4, weights + 8);
    decode_c12(low1.z, low1.w, code_words[3], base4, weights + 12);
    if (unit + ST < KU) prefetch(unit + ST, slot);

    while ((next_escape >> 20) == (uint32_t)unit) {
      if (lane == (int)((next_escape >> 15) & 31)) {
        const uint32_t register_index = (next_escape >> 11) & 15;
        const uint32_t xor_bits =
            (next_escape & 0x7fu) << (8 + ((next_escape >> 10) & 1) * 16);
#pragma unroll
        for (int index = 0; index < 16; ++index)
          if (index == (int)register_index) weights[index] ^= xor_bits;
      }
      next_escape = ++escape_position < 32
          ? __shfl_sync(0xffffffffu, escape, escape_position) : ~0u;
    }

    const uint32_t activation[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter)
      mma_bf16(accumulators[quarter & 1], weights + 4 * quarter,
               activation[2 * quarter], activation[2 * quarter + 1]);
  }

  float4 value = make_float4(
      accumulators[0][0] + accumulators[1][0],
      accumulators[0][1] + accumulators[1][1],
      accumulators[0][2] + accumulators[1][2],
      accumulators[0][3] + accumulators[1][3]);
  reduction[warp][lane] = value;
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int other = 1; other < WK; ++other) {
      const float4 add = reduction[other][lane];
      value.x += add.x; value.y += add.y; value.z += add.z; value.w += add.w;
    }
    if ((lane & 3) == 0) {
      output[tile * 16 + group] = __float2bfloat16_rn(value.x);
      output[tile * 16 + group + 8] = __float2bfloat16_rn(value.z);
    }
  }
}

void codec_e_qkv_m1(
    torch::Tensor low,
    torch::Tensor words,
    torch::Tensor offsets,
    torch::Tensor lut,
    torch::Tensor suffix_bits,
    torch::Tensor bases,
    torch::Tensor x,
    torch::Tensor escapes,
    torch::Tensor output,
    int64_t base4,
    int64_t stages) {
  TORCH_CHECK(x.size(0) == 1 && x.size(1) == 8192, "QKV m1 prototype requires x[1,8192]");
  TORCH_CHECK(output.size(0) == 1 && output.size(1) % 16 == 0, "invalid QKV output");
  const int tiles = (int)output.size(1) / 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (stages == 1)
    codec_e_qkv_m1_kernel<1><<<tiles, 256, 0, stream>>>(reinterpret_cast<const uint4*>(low.data_ptr<uint8_t>()), reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(), suffix_bits.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(escapes.data_ptr<int32_t>()), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), (uint32_t)base4, tiles);
  else if (stages == 2)
    codec_e_qkv_m1_kernel<2><<<tiles, 256, 0, stream>>>(reinterpret_cast<const uint4*>(low.data_ptr<uint8_t>()), reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(), suffix_bits.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(escapes.data_ptr<int32_t>()), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), (uint32_t)base4, tiles);
  else TORCH_CHECK(false, "stages must be 1 or 2");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int ST>
int64_t codec_e_qkv_m1_info_t() {
  auto kernel = codec_e_qkv_m1_kernel<ST>;
  cudaFuncAttributes attributes;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, 0));
  return (int64_t)attributes.numRegs |
         ((int64_t)attributes.localSizeBytes << 10) |
         ((int64_t)attributes.sharedSizeBytes << 30) |
         ((int64_t)active << 50);
}

int64_t codec_e_qkv_m1_info(int64_t stages) {
  if (stages == 1) return codec_e_qkv_m1_info_t<1>();
  if (stages == 2) return codec_e_qkv_m1_info_t<2>();
  TORCH_CHECK(false, "stages must be 1 or 2");
}
"""


def e_pack_fragments(weights: torch.Tensor) -> torch.Tensor:
    n, k = weights.shape
    source = weights.view(n // 16, 16, k // 64, 4, 4, 4)
    low, high = source[:, :8], source[:, 8:]
    fragments = torch.stack(
        (low[..., 0], low[..., 1], high[..., 0], high[..., 1],
         low[..., 2], low[..., 3], high[..., 2], high[..., 3]),
        -1,
    )
    return fragments.permute(0, 2, 4, 1, 3, 5).contiguous()


_OFFSETS = (0, 8, 16, 24, 4, 12, 20, 28)
_SIGNS = (7, 15, 23, 31, 11, 19, 27, 3)


def e_c12_fragments(fragments: torch.Tensor):
    tiles, chunks = fragments.shape[:2]
    device = fragments.device
    bits = fragments.reshape(tiles, chunks, 4, 32, 8).view(torch.int16).to(torch.int32) & 0xFFFF
    exponent, sign = (bits >> 8) & 127, bits >> 15
    histogram = torch.bincount(exponent.flatten().long(), minlength=128)
    base = int(histogram.unfold(0, 8, 1).sum(1).argmax())
    delta = exponent - base
    in_window = (delta >= 0) & (delta < 8)
    delta = torch.where(in_window, delta, 0).long()
    code_words = (
        (delta << torch.tensor(_OFFSETS, device=device)) |
        (sign.long() << torch.tensor(_SIGNS, device=device))
    ).sum(-1)
    code_words = torch.where(code_words >= 2 ** 31, code_words - 2 ** 32, code_words).to(torch.int32)
    low = (bits & 255).to(torch.uint8)
    low0 = low[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(tiles, chunks, 32, 16)
    low1 = low[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(tiles, chunks, 32, 16)
    code_plane = code_words.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(tiles, chunks, 32, 16)
    escaped = (~in_window).nonzero()
    escape_values = exponent[tuple(escaped.t())] ^ base
    escapes = torch.cat((escaped, escape_values[:, None]), dim=1)
    return torch.stack((low0, low1), dim=2).contiguous(), code_plane, base, escapes


def e_escape_lists(escapes: torch.Tensor, tiles: int) -> torch.Tensor:
    output = torch.full((tiles * 8 * 32,), -1, dtype=torch.int64, device=escapes.device)
    if escapes.numel():
        tile, chunk, step, lane, value, xor_value = escapes.t()
        warp_group = tile * 8 + chunk // 16
        sequence = chunk % 16
        order = torch.argsort(warp_group * 4096 + sequence)
        warp_group = warp_group[order]
        entries = (
            (sequence[order] << 20) |
            (lane[order] << 15) |
            ((step[order] * 4 + value[order] // 2) << 11) |
            ((value[order] % 2) << 10) |
            (xor_value[order] & 0x7F)
        )
        rank = torch.arange(len(warp_group), device=escapes.device) - torch.searchsorted(warp_group, warp_group)
        if int(rank.max()) >= 32:
            raise ValueError("a QKV Codec E warp has more than 32 escapes")
        output[warp_group * 32 + rank] = entries
    return torch.where(output >= 2 ** 31, output - 2 ** 32, output).to(torch.int32)


def recover_codec_e_gemv_bits(state):
    """Return independently recovered and original fragment bits."""
    packed = state["_debug_packed"]
    streams = unpack_warp_interleaved(packed)
    if not torch.equal(streams, state["_debug_streams"]):
        raise ValueError("Codec E stream round-trip failed")
    code_plane = state["_debug_code_plane"]
    tiles, chunks = code_plane.shape[:2]
    code_words = code_plane.view(torch.int32).view(tiles, chunks, 32, 4).long() & 0xFFFFFFFF
    code_words = code_words.permute(0, 1, 3, 2)
    device = code_plane.device
    offsets = (code_words[..., None] >> torch.tensor(_OFFSETS, device=device)) & 7
    signs = (code_words[..., None] >> torch.tensor(_SIGNS, device=device)) & 1
    high = state["_debug_base"] + offsets + (signs << 7)
    low_words = state["low"].view(torch.int32).view(tiles, chunks, 2, 32, 4).long() & 0xFFFFFFFF
    low_lanes = torch.stack(
        tuple(low_words[:, :, plane, :, word]
              for plane in range(2) for word in range(4)),
        dim=2,
    ).view(tiles, chunks, 4, 2, 32)
    low_bytes = (
        low_lanes.permute(0, 1, 2, 4, 3)[..., None] >>
        (8 * torch.arange(4, device=device))
    ) & 255
    recovered = (high << 8) | low_bytes.reshape(tiles, chunks, 4, 32, 8)
    escapes = state["_debug_raw_escapes"]
    if escapes.numel():
        tile, chunk, step, lane, value, xor_value = escapes.t()
        recovered[tile, chunk, step, lane, value] ^= xor_value << 8
    original = (state["_debug_fragments"].reshape(tiles, chunks, 4, 32, 8)
                .contiguous().view(torch.int16).to(torch.int32) & 0xFFFF)
    return recovered.to(torch.int32), original


def verify_codec_e_gemv_pack(state) -> bool:
    """Recover every packed fragment bit, independently of the CUDA kernel."""

    recovered, original = recover_codec_e_gemv_bits(state)
    return bool(torch.equal(recovered, original))


_E_GEMV_EXTENSION = None


def e_gemv_extension():
    global _E_GEMV_EXTENSION
    if _E_GEMV_EXTENSION is None:
        digest = hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()[:10]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
        _E_GEMV_EXTENSION = load_inline(
            name=f"codec_e_gemv_{digest}_{arch}",
            cpp_sources=(
                "void codec_e_qkv_m1(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);"
                "int64_t codec_e_qkv_m1_info(int64_t);"
            ),
            cuda_sources=CUDA_SOURCE,
            functions=["codec_e_qkv_m1", "codec_e_qkv_m1_info"],
            extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            verbose=False,
        )
    return _E_GEMV_EXTENSION


def prepare_codec_e_gemv(weights: torch.Tensor):
    if weights.dtype != torch.bfloat16 or weights.shape[1] != 8192 or weights.shape[0] % 16:
        raise ValueError("prototype expects BF16 W[N,8192] with N divisible by 16")
    fragments = e_pack_fragments(weights)
    low, code_plane, base, escapes = e_c12_fragments(fragments)
    tiles, chunks = code_plane.shape[:2]
    streams = code_plane.view(tiles, 8, 16, 32, 16).permute(0, 1, 3, 2, 4).reshape(tiles * 8, 32, 256)
    plan = build_plan(streams)
    packed = pack_warp_interleaved(streams, plan, substreams=4)
    table = {
        "low": low,
        "words": packed.words,
        "offsets": packed.offsets,
        "lut": torch.tensor(plan.lut, dtype=torch.uint8, device=weights.device),
        "suffix_bits": torch.tensor(plan.suffix_bits, dtype=torch.int32, device=weights.device),
        "bases": torch.tensor(plan.bases, dtype=torch.int32, device=weights.device),
        "escapes": e_escape_lists(escapes, tiles),
        "base4": base * 0x01010101,
        "weights": weights,
        "bits_per_weight": 8.0 + (packed.allocated_bits / code_plane.numel()) / 2.0,
        "_debug_code_plane": code_plane,
        "_debug_streams": streams,
        "_debug_packed": packed,
        "_debug_fragments": fragments,
        "_debug_raw_escapes": escapes,
        "_debug_base": base,
    }
    return table


def run_codec_e_gemv(state, x: torch.Tensor, *, stages: int = 2) -> torch.Tensor:
    output = torch.empty((1, state["weights"].shape[0]), dtype=torch.bfloat16, device=x.device)
    e_gemv_extension().codec_e_qkv_m1(
        state["low"], state["words"], state["offsets"].reshape(-1),
        state["lut"], state["suffix_bits"], state["bases"], x,
        state["escapes"], output, state["base4"], stages,
    )
    return output


def codec_e_gemv_resource_info(stages: int):
    packed = int(e_gemv_extension().codec_e_qkv_m1_info(stages))
    return {
        "registers": packed & ((1 << 10) - 1),
        "local_bytes": (packed >> 10) & ((1 << 20) - 1),
        "shared_bytes": (packed >> 30) & ((1 << 20) - 1),
        "active_ctas_per_sm": packed >> 50,
    }
