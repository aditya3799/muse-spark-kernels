"""Local microbenchmark for the warp-private page-1 KV gather design.

This does not contact the OpenKernel service.  Each warp streams a disjoint
sequence of 32-key groups.  For every group, every lane issues four 16-byte K
loads and eight 8-byte V loads before consuming any result.  This reproduces
the memory-level-parallelism target of the proposed register-fragment kernel
without conflating it with MMA, softmax, or split reduction costs.
"""

from __future__ import annotations

import argparse
import os
from statistics import median

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

struct U32x4 { uint32_t x, y, z, w; };
struct U32x2 { uint32_t x, y; };

__device__ __forceinline__ U32x4 load_nc_128(const void* ptr) {
  U32x4 out;
  asm volatile(
      "ld.global.nc.L1::no_allocate.v4.u32 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(ptr));
  return out;
}

__device__ __forceinline__ U32x2 load_nc_64(const void* ptr) {
  U32x2 out;
  asm volatile(
      "ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
      : "=r"(out.x), "=r"(out.y)
      : "l"(ptr));
  return out;
}

__global__ __launch_bounds__(256, 2)
void warp_private_gather_kernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    uint32_t* __restrict__ checksums,
    int context,
    int splits) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int task = blockIdx.x;
  const int split = task % splits;
  const int kv_head = (task / splits) & 1;
  const int batch = task / (splits * 2);
  const int begin = static_cast<int>((static_cast<long long>(context) * split) / splits);
  const int end = static_cast<int>((static_cast<long long>(context) * (split + 1)) / splits);
  const int groups = (end - begin + 31) >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  uint32_t sink = 0;

  for (int group = warp; group < groups; group += 8) {
    const int pos0 = begin + group * 32;
    const int lane_pos = pos0 + lane;
    const int lane_page = lane_pos < end
        ? page_table[static_cast<long long>(batch) * context + lane_pos]
        : page_table[static_cast<long long>(batch) * context + begin];

    // Four independent K rows.  The four lanes of each quad cover one 64-B
    // row with aligned 16-B loads.
    const int kp0 = __shfl_sync(0xffffffffu, lane_page, g + 0);
    const int kp1 = __shfl_sync(0xffffffffu, lane_page, g + 8);
    const int kp2 = __shfl_sync(0xffffffffu, lane_page, g + 16);
    const int kp3 = __shfl_sync(0xffffffffu, lane_page, g + 24);
    const long long kh0 = (static_cast<long long>(kp0) * 2 + kv_head) * 64 + t * 16;
    const long long kh1 = (static_cast<long long>(kp1) * 2 + kv_head) * 64 + t * 16;
    const long long kh2 = (static_cast<long long>(kp2) * 2 + kv_head) * 64 + t * 16;
    const long long kh3 = (static_cast<long long>(kp3) * 2 + kv_head) * 64 + t * 16;

    // Eight independent V rows.  Lanes with equal t collectively cover the
    // 64-B row in eight aligned 8-B pieces (one piece per g).
    const int vp0 = __shfl_sync(0xffffffffu, lane_page, 2 * t + 0);
    const int vp1 = __shfl_sync(0xffffffffu, lane_page, 2 * t + 1);
    const int vp2 = __shfl_sync(0xffffffffu, lane_page, 8 + 2 * t + 0);
    const int vp3 = __shfl_sync(0xffffffffu, lane_page, 8 + 2 * t + 1);
    const int vp4 = __shfl_sync(0xffffffffu, lane_page, 16 + 2 * t + 0);
    const int vp5 = __shfl_sync(0xffffffffu, lane_page, 16 + 2 * t + 1);
    const int vp6 = __shfl_sync(0xffffffffu, lane_page, 24 + 2 * t + 0);
    const int vp7 = __shfl_sync(0xffffffffu, lane_page, 24 + 2 * t + 1);
    const long long vd = g * 8;
    const long long vh0 = (static_cast<long long>(vp0) * 2 + kv_head) * 64 + vd;
    const long long vh1 = (static_cast<long long>(vp1) * 2 + kv_head) * 64 + vd;
    const long long vh2 = (static_cast<long long>(vp2) * 2 + kv_head) * 64 + vd;
    const long long vh3 = (static_cast<long long>(vp3) * 2 + kv_head) * 64 + vd;
    const long long vh4 = (static_cast<long long>(vp4) * 2 + kv_head) * 64 + vd;
    const long long vh5 = (static_cast<long long>(vp5) * 2 + kv_head) * 64 + vd;
    const long long vh6 = (static_cast<long long>(vp6) * 2 + kv_head) * 64 + vd;
    const long long vh7 = (static_cast<long long>(vp7) * 2 + kv_head) * 64 + vd;

    // Keep all twelve loads independent until every request has been issued.
    const U32x4 k0 = load_nc_128(k_cache + kh0);
    const U32x4 k1 = load_nc_128(k_cache + kh1);
    const U32x4 k2 = load_nc_128(k_cache + kh2);
    const U32x4 k3 = load_nc_128(k_cache + kh3);
    const U32x2 v0 = load_nc_64(v_cache + vh0);
    const U32x2 v1 = load_nc_64(v_cache + vh1);
    const U32x2 v2 = load_nc_64(v_cache + vh2);
    const U32x2 v3 = load_nc_64(v_cache + vh3);
    const U32x2 v4 = load_nc_64(v_cache + vh4);
    const U32x2 v5 = load_nc_64(v_cache + vh5);
    const U32x2 v6 = load_nc_64(v_cache + vh6);
    const U32x2 v7 = load_nc_64(v_cache + vh7);

    sink ^= k0.x ^ k0.y ^ k0.z ^ k0.w;
    sink ^= k1.x ^ k1.y ^ k1.z ^ k1.w;
    sink ^= k2.x ^ k2.y ^ k2.z ^ k2.w;
    sink ^= k3.x ^ k3.y ^ k3.z ^ k3.w;
    sink ^= v0.x ^ v0.y ^ v1.x ^ v1.y ^ v2.x ^ v2.y ^ v3.x ^ v3.y;
    sink ^= v4.x ^ v4.y ^ v5.x ^ v5.y ^ v6.x ^ v6.y ^ v7.x ^ v7.y;
  }

  // One register result per thread.  No warp reduction or atomics are in the
  // measured loop or epilogue.
  checksums[task * 256 + threadIdx.x] = sink;
}

__global__ __launch_bounds__(512, 2)
void scalar_gather_kernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    uint32_t* __restrict__ checksums,
    int context,
    int splits) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int task = blockIdx.x;
  const int split = task % splits;
  const int kv_head = (task / splits) & 1;
  const int batch = task / (splits * 2);
  const int begin = static_cast<int>((static_cast<long long>(context) * split) / splits);
  const int end = static_cast<int>((static_cast<long long>(context) * (split + 1)) / splits);
  uint32_t sink = 0;

  for (int pos = begin + warp; pos < end; pos += 16) {
    int page = 0;
    if (lane == 0) {
      page = page_table[static_cast<long long>(batch) * context + pos];
    }
    page = __shfl_sync(0xffffffffu, page, 0);
    const long long offset =
        (static_cast<long long>(page) * 2 + kv_head) * 64 + lane * 2;
    const uint16_t kval = *reinterpret_cast<const uint16_t*>(k_cache + offset);
    const uint16_t vval = *reinterpret_cast<const uint16_t*>(v_cache + offset);
    sink ^= static_cast<uint32_t>(kval) | (static_cast<uint32_t>(vval) << 16);
  }

  checksums[task * 512 + threadIdx.x] = sink;
}

__global__ __launch_bounds__(256)
void contiguous_copy_kernel(
    const uint4* __restrict__ source,
    uint4* __restrict__ destination,
    long long words) {
  for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < words;
       index += static_cast<long long>(gridDim.x) * blockDim.x) {
    destination[index] = source[index];
  }
}

torch::Tensor run_gather(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor output,
    int64_t splits) {
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && page_table.is_cuda() && output.is_cuda());
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && page_table.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(page_table.scalar_type() == at::kInt);
  const int batch = page_table.size(0);
  const int context = page_table.size(1);
  const int blocks = batch * 2 * static_cast<int>(splits);
  TORCH_CHECK(output.scalar_type() == at::kInt && output.numel() >= blocks * 256ll);
  auto stream = at::cuda::getCurrentCUDAStream();
  warp_private_gather_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()),
      page_table.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()),
      context,
      static_cast<int>(splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor run_scalar_gather(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor output,
    int64_t splits) {
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && page_table.is_cuda() && output.is_cuda());
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && page_table.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(page_table.scalar_type() == at::kInt);
  const int batch = page_table.size(0);
  const int context = page_table.size(1);
  const int blocks = batch * 2 * static_cast<int>(splits);
  TORCH_CHECK(output.scalar_type() == at::kInt && output.numel() >= blocks * 512ll);
  auto stream = at::cuda::getCurrentCUDAStream();
  scalar_gather_kernel<<<blocks, 512, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_cache.data_ptr()),
      page_table.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()),
      context,
      static_cast<int>(splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor run_copy(torch::Tensor source, torch::Tensor destination) {
  TORCH_CHECK(source.is_cuda() && destination.is_cuda());
  TORCH_CHECK(source.is_contiguous() && destination.is_contiguous());
  TORCH_CHECK(source.scalar_type() == at::kByte && destination.scalar_type() == at::kByte);
  TORCH_CHECK(source.numel() == destination.numel() && source.numel() % 16 == 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  contiguous_copy_kernel<<<256, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(source.data_ptr<uint8_t>()),
      reinterpret_cast<uint4*>(destination.data_ptr<uint8_t>()),
      source.numel() / 16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return destination;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_gather", &run_gather);
  m.def("run_scalar_gather", &run_scalar_gather);
  m.def("run_copy", &run_copy);
}
"""


def build():
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        os.path.join(os.path.dirname(__file__), ".torch_extensions"),
    )
    return load_inline(
        name="okc_warp_private_gather",
        cpp_sources="",
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        with_cuda=True,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=80_000)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    module = build()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{''.join(map(str, torch.cuda.get_device_capability()))})")

    flush = torch.empty((64 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
    for batch in args.batches:
        pages = batch * args.ctx
        # Byte tensors have the same physical layout as the challenge FP8 cache.
        k = torch.randint(0, 256, (pages, 1, 2, 64), device="cuda", dtype=torch.uint8)
        v = torch.randint(0, 256, (pages, 1, 2, 64), device="cuda", dtype=torch.uint8)
        table = torch.randperm(pages, device="cuda", dtype=torch.int64).to(torch.int32).view(batch, args.ctx)
        splits = 8 if batch == 16 else 4

        useful_bytes = batch * args.ctx * 2 * 2 * 64
        blocks = batch * 2 * splits
        wide_output = torch.empty((blocks, 256), device="cuda", dtype=torch.int32)
        scalar_output = torch.empty((blocks, 512), device="cuda", dtype=torch.int32)
        copy_source = torch.randint(0, 256, (useful_bytes,), device="cuda", dtype=torch.uint8)
        copy_destination = torch.empty_like(copy_source)
        results = {}
        for name, function in (
            ("scalar-u16", lambda: module.run_scalar_gather(k, v, table, scalar_output, splits)),
            ("wide-queued", lambda: module.run_gather(k, v, table, wide_output, splits)),
            ("contiguous", lambda: module.run_copy(copy_source, copy_destination)),
            ("torch-copy", lambda: copy_destination.copy_(copy_source)),
        ):
            for _ in range(args.warmup):
                function()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            result = None
            for _ in range(args.repeats):
                flush.zero_()
                start.record()
                result = function()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0)
            usec = median(samples)
            results[name] = usec
            payload_tb_s = useful_bytes / (usec * 1.0e6)
            traffic_tb_s = payload_tb_s * (2 if name == "contiguous" else 1)
            marker = int(result.flatten()[0])
            print(f"B={batch:2d} ctx={args.ctx:6d} {name:11s}: {usec:9.2f} us  "
                  f"payload_BW={payload_tb_s:6.3f} TB/s  traffic_BW={traffic_tb_s:6.3f} TB/s  marker={marker}")
        print(f"  wide/scalar speedup: {results['scalar-u16'] / results['wide-queued']:.3f}x")


if __name__ == "__main__":
    main()
