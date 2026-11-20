"""CUDA correctness and cold-L2 throughput probe for Codec E.

This is deliberately independent of ``kernel_fn``.  It answers the first
promotion question: can a spill-free streaming decoder recover the existing
C12 high-code bytes fast enough to justify integration into GVCU?
"""

from __future__ import annotations

import argparse
import hashlib
import statistics

import torch
from torch.utils.cpp_extension import load_inline

from codec_e import build_plan, c12_pair_symbols, pack_warp_interleaved


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t load_nc(const uint32_t* pointer) {
  uint32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
#else
  value = __ldg(pointer);
#endif
  return value;
}

__global__ void codec_e_decode_kernel(
    const uint32_t* __restrict__ words,
    const int32_t* __restrict__ offsets,
    uint8_t* __restrict__ output,
    const int32_t* __restrict__ suffix_bits,
    const int32_t* __restrict__ bases,
    const uint8_t* __restrict__ lut,
    int streams,
    int symbols_per_stream) {
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

  const int stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) return;
  const uint32_t* input = words + offsets[stream];
  uint64_t buffer = (uint64_t)load_nc(input) | ((uint64_t)load_nc(input + 1) << 32);
  int available = 64;
  int next_word = 2;
  uint8_t* destination = output + (size_t)stream * symbols_per_stream;

#pragma unroll 1
  for (int index = 0; index < symbols_per_stream; ++index) {
    const uint32_t group = (uint32_t)buffer & 7u;
    const uint32_t width = shared_suffix_bits[group];
    const uint32_t suffix = ((uint32_t)(buffer >> 3)) & ((1u << width) - 1u);
    destination[index] = shared_lut[shared_bases[group] + suffix];
    const int consumed = 3 + (int)width;
    buffer >>= consumed;
    available -= consumed;
    if (available < 11 && index + 1 < symbols_per_stream) {
      buffer |= (uint64_t)load_nc(input + next_word++) << available;
      available += 32;
    }
  }
}

__global__ void copy_symbols_kernel(
    const uint4* __restrict__ source,
    uint4* __restrict__ destination,
    int vectors_per_stream,
    int streams) {
  const int stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) return;
  source += (size_t)stream * vectors_per_stream;
  destination += (size_t)stream * vectors_per_stream;
#pragma unroll 1
  for (int index = 0; index < vectors_per_stream; ++index)
    destination[index] = __ldg(source + index);
}

template <int SUBSTREAMS>
__global__ void codec_e_hash_kernel(
    const uint32_t* __restrict__ words,
    const int32_t* __restrict__ offsets,
    uint32_t* __restrict__ output,
    const int32_t* __restrict__ suffix_bits,
    const int32_t* __restrict__ bases,
    const uint8_t* __restrict__ lut,
    int streams,
    int symbols_per_stream) {
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

  const int stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) return;
  const int warp = stream >> 5, lane = stream & 31;
  uint64_t buffers[SUBSTREAMS];
  int available[SUBSTREAMS], next_word[SUBSTREAMS];
#pragma unroll
  for (int substream = 0; substream < SUBSTREAMS; ++substream) {
    const int base = offsets[warp * SUBSTREAMS + substream];
    buffers[substream] = (uint64_t)load_nc(words + base + lane) |
                         ((uint64_t)load_nc(words + base + 32 + lane) << 32);
    available[substream] = 64;
    next_word[substream] = 2;
  }

  uint32_t hash = 2166136261u;
#pragma unroll 1
  for (int position = 0; position < symbols_per_stream / SUBSTREAMS; ++position) {
#pragma unroll
    for (int substream = 0; substream < SUBSTREAMS; ++substream) {
      uint64_t buffer = buffers[substream];
      const uint32_t group = (uint32_t)buffer & 7u;
      const uint32_t width = shared_suffix_bits[group];
      const uint32_t suffix = ((uint32_t)(buffer >> 3)) & ((1u << width) - 1u);
      hash = (hash ^ shared_lut[shared_bases[group] + suffix]) * 16777619u;
      const int consumed = 3 + (int)width;
      buffer >>= consumed;
      available[substream] -= consumed;
      if (available[substream] < 11 && position + 1 < symbols_per_stream / SUBSTREAMS) {
        const int base = offsets[warp * SUBSTREAMS + substream];
        buffer |= (uint64_t)load_nc(words + base + next_word[substream] * 32 + lane)
                  << available[substream];
        available[substream] += 32;
        ++next_word[substream];
      }
      buffers[substream] = buffer;
    }
  }
  output[stream] = hash;
}

__global__ void raw_hash_kernel(
    const uint4* __restrict__ source,
    uint32_t* __restrict__ output,
    int streams,
    int units) {
  const int stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) return;
  const int warp = stream >> 5, lane = stream & 31;
  uint32_t hash = 2166136261u;
#pragma unroll 1
  for (int unit = 0; unit < units; ++unit) {
    const uint4 value = __ldg(source + ((size_t)warp * units + unit) * 32 + lane);
    const uint32_t words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
    for (int word = 0; word < 4; ++word)
#pragma unroll
      for (int byte = 0; byte < 4; ++byte)
        hash = (hash ^ ((words[word] >> (8 * byte)) & 0xffu)) * 16777619u;
  }
  output[stream] = hash;
}

void codec_e_decode(
    torch::Tensor words,
    torch::Tensor offsets,
    torch::Tensor output,
    torch::Tensor suffix_bits,
    torch::Tensor bases,
    torch::Tensor lut) {
  TORCH_CHECK(words.is_cuda() && offsets.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(words.scalar_type() == at::kInt && offsets.scalar_type() == at::kInt, "int32 words/offsets required");
  TORCH_CHECK(output.scalar_type() == at::kByte && output.dim() == 2, "2-D uint8 output required");
  const int streams = (int)output.size(0);
  const int symbols = (int)output.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  codec_e_decode_kernel<<<(streams + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()),
      offsets.data_ptr<int32_t>(),
      output.data_ptr<uint8_t>(),
      suffix_bits.data_ptr<int32_t>(),
      bases.data_ptr<int32_t>(),
      lut.data_ptr<uint8_t>(),
      streams,
      symbols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_symbols(torch::Tensor source, torch::Tensor output) {
  TORCH_CHECK(source.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(source.scalar_type() == at::kByte && source.sizes() == output.sizes(), "matching uint8 tensors required");
  TORCH_CHECK(source.dim() == 2 && source.size(1) % 16 == 0, "streams must be a multiple of 16 bytes");
  const int streams = (int)source.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  copy_symbols_kernel<<<(streams + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const uint4*>(source.data_ptr<uint8_t>()),
      reinterpret_cast<uint4*>(output.data_ptr<uint8_t>()),
      (int)source.size(1) / 16,
      streams);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void codec_e_hash(
    torch::Tensor words,
    torch::Tensor offsets,
    torch::Tensor output,
    torch::Tensor suffix_bits,
    torch::Tensor bases,
    torch::Tensor lut,
    int64_t symbols,
    int64_t substreams) {
  TORCH_CHECK(words.is_cuda() && offsets.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(output.scalar_type() == at::kInt, "int32 hash output required");
  const int streams = (int)output.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((streams + 127) / 128), block(128);
  if (substreams == 1)
    codec_e_hash_kernel<1><<<grid, block, 0, stream>>>(reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(), reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), suffix_bits.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(), streams, (int)symbols);
  else if (substreams == 2)
    codec_e_hash_kernel<2><<<grid, block, 0, stream>>>(reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(), reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), suffix_bits.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(), streams, (int)symbols);
  else if (substreams == 4)
    codec_e_hash_kernel<4><<<grid, block, 0, stream>>>(reinterpret_cast<const uint32_t*>(words.data_ptr<int32_t>()), offsets.data_ptr<int32_t>(), reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), suffix_bits.data_ptr<int32_t>(), bases.data_ptr<int32_t>(), lut.data_ptr<uint8_t>(), streams, (int)symbols);
  else TORCH_CHECK(false, "substreams must be 1, 2, or 4");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void raw_hash(torch::Tensor source, torch::Tensor output) {
  TORCH_CHECK(source.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(source.scalar_type() == at::kByte && source.dim() == 4 && source.size(2) == 32 && source.size(3) == 16, "[warp,unit,32,16] byte source required");
  const int streams = (int)source.size(0) * 32;
  TORCH_CHECK(output.scalar_type() == at::kInt && output.numel() == streams, "one int32 hash per stream required");
  auto stream = at::cuda::getCurrentCUDAStream();
  raw_hash_kernel<<<(streams + 127) / 128, 128, 0, stream>>>(reinterpret_cast<const uint4*>(source.data_ptr<uint8_t>()), reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()), streams, (int)source.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SUBSTREAMS>
int64_t codec_e_hash_info_t() {
  auto kernel = codec_e_hash_kernel<SUBSTREAMS>;
  cudaFuncAttributes attributes;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128, 0));
  return (int64_t)attributes.numRegs |
         ((int64_t)attributes.localSizeBytes << 10) |
         ((int64_t)attributes.sharedSizeBytes << 30) |
         ((int64_t)active << 50);
}

int64_t codec_e_hash_info(int64_t substreams) {
  if (substreams == 1) return codec_e_hash_info_t<1>();
  if (substreams == 2) return codec_e_hash_info_t<2>();
  if (substreams == 4) return codec_e_hash_info_t<4>();
  TORCH_CHECK(false, "substreams must be 1, 2, or 4");
}
"""


_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        digest = hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()[:10]
        _EXTENSION = load_inline(
            name=f"codec_e_probe_{digest}",
            cpp_sources=(
                "void codec_e_decode(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);"
                "void copy_symbols(torch::Tensor,torch::Tensor);"
                "void codec_e_hash(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);"
                "void raw_hash(torch::Tensor,torch::Tensor);"
                "int64_t codec_e_hash_info(int64_t);"
            ),
            cuda_sources=CUDA_SOURCE,
            functions=["codec_e_decode", "copy_symbols", "codec_e_hash", "raw_hash", "codec_e_hash_info"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXTENSION


def _flatten_substreams(symbols: torch.Tensor, substreams: int) -> torch.Tensor:
    per_substream = symbols.shape[-1] // substreams
    return torch.stack([symbols[..., lane::substreams] for lane in range(substreams)], dim=-2).reshape(-1, per_substream)


def decode_gpu(packed):
    device = packed.words.device
    suffix_bits = torch.tensor(packed.plan.suffix_bits, dtype=torch.int32, device=device)
    bases = torch.tensor(packed.plan.bases, dtype=torch.int32, device=device)
    lut = torch.tensor(packed.plan.lut, dtype=torch.uint8, device=device)
    output = torch.empty(
        (packed.offsets.numel(), packed.symbols_per_stream // packed.substreams),
        dtype=torch.uint8,
        device=device,
    )
    extension().codec_e_decode(packed.words, packed.offsets.reshape(-1), output, suffix_bits, bases, lut)
    return output, suffix_bits, bases, lut


def hash_gpu(packed):
    device = packed.words.device
    suffix_bits = torch.tensor(packed.plan.suffix_bits, dtype=torch.int32, device=device)
    bases = torch.tensor(packed.plan.bases, dtype=torch.int32, device=device)
    lut = torch.tensor(packed.plan.lut, dtype=torch.uint8, device=device)
    output = torch.empty(packed.offsets.numel() // packed.substreams * 32, dtype=torch.int32, device=device)
    extension().codec_e_hash(
        packed.words,
        packed.offsets.reshape(-1),
        output,
        suffix_bits,
        bases,
        lut,
        packed.symbols_per_stream,
        packed.substreams,
    )
    return output, suffix_bits, bases, lut


def _capture(function):
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def _cold_times(graph, flush: torch.Tensor, sink: torch.Tensor, repetitions: int):
    times = []
    for _ in range(repetitions):
        torch.sum(flush, dim=0, out=sink)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(begin.elapsed_time(end) * 1000.0)
    return times


def representative_symbols(rows: int, k: int, stream_symbols: int, device: str):
    generator = torch.Generator(device=device).manual_seed(5151)
    weights = (torch.randn((rows, k), generator=generator, device=device) / (k ** 0.5)).to(torch.bfloat16)
    _low, symbols, _base, _escapes, _original = c12_pair_symbols(weights)
    flat = symbols.reshape(-1)
    usable = flat.numel() // stream_symbols * stream_symbols
    return flat[:usable].reshape(-1, stream_symbols).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2304)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--stream-symbols", type=int, default=256,
                        help="high-code bytes consumed sequentially by one lane")
    parser.add_argument("--repetitions", type=int, default=21)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = "cuda"
    symbols = representative_symbols(args.rows, args.k, args.stream_symbols, device)
    usable_streams = symbols.shape[0] // 32 * 32
    symbols = symbols[:usable_streams].reshape(-1, 32, args.stream_symbols).contiguous()
    plan = build_plan(symbols)
    print(f"device={torch.cuda.get_device_name()} symbols={symbols.numel():,} streams={symbols.numel() // args.stream_symbols:,}")
    print(f"suffix_bits={plan.suffix_bits} bases={plan.bases}")
    flush = torch.ones((64 * 1024 * 1024 // 1024, 1024), dtype=torch.float32, device=device)
    sink = torch.empty(1024, dtype=torch.float32, device=device)

    raw_input = symbols.view(symbols.shape[0], 32, -1, 16).permute(0, 2, 1, 3).contiguous()
    raw_output = torch.empty(symbols.shape[0] * 32, dtype=torch.int32, device=device)
    raw_graph = _capture(lambda: extension().raw_hash(raw_input, raw_output))
    raw_times = _cold_times(raw_graph, flush, sink, args.repetitions)

    for substreams in (1, 2, 4):
        packed = pack_warp_interleaved(symbols, plan, substreams=substreams)
        output, suffix_bits, bases, lut = hash_gpu(packed)
        torch.cuda.synchronize()
        if not torch.equal(output, raw_output):
            mismatch = int((output != raw_output).nonzero()[0])
            raise AssertionError(f"GPU Codec E hash mismatch in stream {mismatch}")
        # Keep the table tensors alive in the captured closure.
        graph = _capture(lambda p=packed, o=output, sb=suffix_bits, b=bases, l=lut:
                         extension().codec_e_hash(p.words, p.offsets.reshape(-1), o, sb, b, l,
                                                  p.symbols_per_stream, p.substreams))
        times = _cold_times(graph, flush, sink, args.repetitions)
        raw_median = statistics.median(raw_times)
        median = statistics.median(times)
        allocated = packed.allocated_bits / symbols.numel()
        payload = packed.bits_per_symbol
        # A code byte represents two weights; add the unchanged eight low bits/weight.
        total_payload = 8.0 + payload / 2.0
        total_allocated = 8.0 + allocated / 2.0
        info = int(extension().codec_e_hash_info(substreams))
        resources = {
            "registers": info & ((1 << 10) - 1),
            "local_bytes": (info >> 10) & ((1 << 20) - 1),
            "shared_bytes": (info >> 30) & ((1 << 20) - 1),
            "active_ctas_per_sm": info >> 50,
        }
        print(
            f"substreams={substreams} payload={total_payload:.5f} bits/weight "
            f"allocated={total_allocated:.5f} raw_copy={raw_median:.3f} us "
            f"codec={median:.3f} us ratio={median / raw_median:.4f} resources={resources}"
        )


if __name__ == "__main__":
    main()
