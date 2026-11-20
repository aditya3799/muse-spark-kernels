"""Non-scored source and encoding validation for v143's packed escapes."""

import random
import sys
import types

import torch
from numba.cuda.cudadrv import nvrtc


MOCK_NAMES = [
    "sglang", "sglang.srt", "sglang.srt.layers",
    "sglang.srt.layers.quantization",
    "sglang.srt.layers.quantization.unquant",
]
MOCKS = {name: types.ModuleType(name) for name in MOCK_NAMES}
MOCKS[MOCK_NAMES[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
sys.modules.update(MOCKS)

import submission_escape_radix_v143 as candidate


PREAMBLE = r"""
#include <cuda_bf16.h>
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) -
         __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}

__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
"""

INSTANTIATIONS = r"""
template __global__ void cgemv_radix_kernel<2, 256, 8192>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
template __global__ void cgemv_radix_kernel<2, 512, 8192>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
template __global__ void cgemv_radix_kernel<2, 64, 1024>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
template __global__ void cgemv_radix_kernel<2, 32, 1024>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
template __global__ void cgemv_radix_kernel<2, 128, 4096>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
template __global__ void cgemv_radix_kernel<2, 256, 4096>(
    const uint8_t*, const uint8_t*, const __nv_bfloat16*, const uint32_t*,
    int, __nv_bfloat16*, uint32_t);
"""


def validate_encoding():
    rng = random.Random(143)
    ncta, pmax = 128, 33
    tab = torch.zeros((ncta, pmax, 2), dtype=torch.int32)
    expected = []
    for cta in range(ncta):
        count = rng.randrange(pmax)
        tab[cta, 0, 0] = count
        for slot in range(1, count + 1):
            row = rng.randrange(2)
            k = rng.randrange(8192)
            bits = rng.randrange(65536)
            tab[cta, slot, 0] = (row << 16) | k
            tab[cta, slot, 1] = bits
            expected.append((cta, slot, row, k, bits))
    packed = candidate._v143_pack_table(tab)
    assert torch.equal(packed[:, 0], tab[:, 0, 0])
    for cta, slot, row, k, bits in expected:
        word = int(packed[cta, slot])
        assert ((word >> 29) & 1, (word >> 16) & 0x1FFF,
                word & 0xFFFF) == (row, k, bits)
    assert packed.numel() * packed.element_size() * 2 == tab.numel() * tab.element_size()
    print("encoding round-trip OK", len(expected), "leaves",
          tab.numel() * tab.element_size(), "->",
          packed.numel() * packed.element_size(), "bytes")


def validate_cuda_source():
    kernel = candidate._v143_radix_kernel_source()
    assert "const uint32_t* __restrict__ etab" in kernel
    assert "en.x" not in kernel and "en.y" not in kernel
    ptx, log = nvrtc.compile(
        PREAMBLE + kernel + INSTANTIATIONS, "v143_radix.cu", (9, 0))
    assert "cgemv_radix_kernel" in ptx
    print("NVRTC compute_90 compile OK", len(ptx), "PTX bytes",
          "log_bytes", len(log))


if __name__ == "__main__":
    validate_encoding()
    validate_cuda_source()
