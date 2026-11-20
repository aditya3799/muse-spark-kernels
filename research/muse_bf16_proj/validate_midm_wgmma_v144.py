"""Non-scored geometry and compute_90a source validation for v144."""

import os
import sys
import types

from numba.cuda.cudadrv import nvrtc


MOCK_NAMES = [
    "sglang", "sglang.srt", "sglang.srt.layers",
    "sglang.srt.layers.quantization",
    "sglang.srt.layers.quantization.unquant",
]
MOCKS = {name: types.ModuleType(name) for name in MOCK_NAMES}
MOCKS[MOCK_NAMES[-1]]._bf16_gemm_dispatch_impl = lambda *args, **kwargs: None
sys.modules.update(MOCKS)

import submission_midm_wgmma_v144 as candidate


SHAPES = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
}


INSTANTIATIONS = r"""
template __global__ void skinny_tma_kernel<2, 8, 2, 4, 4, true, false, true, __nv_bfloat16>(
    const uint4*, const __nv_bfloat16*, __nv_bfloat16*, float*, int*,
    const int2*, int, uint32_t, int, int, int, int, int, int, int, int);
template __global__ void skinny_tma_kernel<2, 4, 4, 4, 4, true, false, true, __nv_bfloat16>(
    const uint4*, const __nv_bfloat16*, __nv_bfloat16*, float*, int*,
    const int2*, int, uint32_t, int, int, int, int, int, int, int, int);
template __global__ void skinny_tma_kernel<4, 8, 1, 8, 4, true, false, true, __nv_bfloat16>(
    const uint4*, const __nv_bfloat16*, __nv_bfloat16*, float*, int*,
    const int2*, int, uint32_t, int, int, int, int, int, int, int, int);
template __global__ void skinny_tma_kernel<4, 4, 4, 4, 4, true, false, true, __nv_bfloat16>(
    const uint4*, const __nv_bfloat16*, __nv_bfloat16*, float*, int*,
    const int2*, int, uint32_t, int, int, int, int, int, int, int, int);
template __global__ void skinny_tma_kernel<4, 16, 1, 4, 4, true, false, true, __nv_bfloat16>(
    const uint4*, const __nv_bfloat16*, __nv_bfloat16*, float*, int*,
    const int2*, int, uint32_t, int, int, int, int, int, int, int, int);
"""


def validate_geometries():
    dispatch = candidate._tma_srcs(True)
    summaries = []
    for layer, m in candidate._V144_TARGETS:
        n, k = SHAPES[layer]
        splits, rg, wsn, ks = candidate._V144_CONFIGS[(layer, m)]
        mt = candidate._mt_key(m)
        assert rg % 4 == 0
        assert n % (16 * rg) == 0
        assert k % splits == 0
        assert (k // splits) % 64 == 0
        assert (rg * wsn + 1) * 32 <= 1024
        expected = (
            f"mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} "
            "&& comp == 1 && out_f32 == false"
        )
        assert expected in dispatch
        launch = (
            f"launch_tma<{mt}, {rg}, {wsn}, {ks}, 4, true, false, true, "
            "__nv_bfloat16>"
        )
        assert launch in dispatch
        grid = (n // (16 * rg)) * splits
        summaries.append((layer, m, (splits, rg, wsn, ks), grid))
    print("geometry/dispatch validation OK", summaries)


def compile_compute_90a():
    common = candidate.CUDA_SRC
    common = common.replace("#include <ATen/core/Tensor.h>\n", "")
    common = common.replace("#include <ATen/cuda/CUDAContext.h>\n", "")
    common = common.replace("#include <c10/cuda/CUDAException.h>\n", "")
    common = common.replace(
        "#include <stdint.h>",
        "typedef unsigned int uint32_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "typedef unsigned char uint8_t;\n"
        "namespace std { template <bool B, class T = void> struct enable_if {}; "
        "template <class T> struct enable_if<true, T> { typedef T type; }; }",
    )
    common = common.replace("@@MBAR@@", candidate.MBAR_HELPERS)
    common = common.replace("@@BODY@@", candidate.WG_HELPERS)
    begin = candidate.TMA_SRC.index(
        "template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP")
    end = candidate.TMA_SRC.index(
        "template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP",
        begin + 10,
    )
    kernel = candidate.TMA_SRC[begin:end]
    source = (common + candidate.wgmma_defs([16, 32]) + kernel +
              INSTANTIATIONS)

    compiler = nvrtc.NVRTC()
    program = compiler.create_program(source, "v144_midm_wgmma.cu")
    include = f"-I{nvrtc.config.CUDA_INCLUDE_PATH}"
    options = ["--gpu-architecture=compute_90a", include,
               "--std=c++17", "-rdc", "true"]
    error = compiler.compile_program(program, options)
    log = compiler.get_compile_log(program)
    if error:
        raise RuntimeError("NVRTC compute_90a failure:\n" + log)
    ptx = compiler.get_ptx(program)
    assert "skinny_tma_kernel" in ptx
    assert "wgmma.mma_async" in ptx
    print("NVRTC compute_90a compile OK", len(ptx), "PTX bytes",
          "log_bytes", len(log))
    if log:
        print(log.strip())


if __name__ == "__main__":
    validate_geometries()
    compile_compute_90a()
