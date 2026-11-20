"""Compile-only feasibility probe for full-token lossless-C12 WGMMA tiles.

This imports the already validated v55 packer/kernel body, adds the Hopper
m64n128 and m64n256 register-A WGMMA instructions, and compiles three candidate
geometries.  It does not modify or dispatch the production submission.
"""

from pathlib import Path
import importlib.util
import hashlib

from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parent
V55 = ROOT / "submission_unpooler128_c12_v55.py"

spec = importlib.util.spec_from_file_location("v55_uwide_probe", V55)
v55 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v55)


INSTANCES = [
    # BN, R, WK, prefetch, activation stages, C12, launch-bounds capacity
    (128, 64, 4, 2, 2, True, 1),
    (256, 64, 2, 2, 2, True, 1),
    (256, 32, 2, 2, 2, True, 1),
]


def cuda_source():
    cases = []
    attrs = []
    for i, inst in enumerate(INSTANCES):
        bn, r, wk, pf, q, compressed, cap = inst
        params = f"{bn},{r},{wk},{pf},{q},{str(compressed).lower()},{cap}"
        cases.append(
            f"case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,"
            "(void*)out,m,n,k,split,tab,(cudaStream_t)st);"
        )
        attrs.append(
            f"case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;"
        )
    core = v55._u32_CUDA_CORE.replace(
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));",
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));",
    )
    return (
        v55._u32_CUDA_PREFIX
        + "\n".join(v55._u32_mma(n) for n in (16, 32, 64, 128, 256))
        + core
        + v55._u32_CUDA_MAP
        + v55._u32_CUDA_HOST.replace("DISPATCH", "\n".join(cases)).replace(
            "INFO", "\n".join(attrs)
        )
    )


src = cuda_source()
name = "probe_uwide_c12_" + hashlib.sha256(src.encode()).hexdigest()[:12]
load_inline(
    name,
    v55._u32_CPP,
    cuda_sources=src,
    functions=["launch", "info"],
    with_pytorch_error_handling=False,
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--ptxas-options=-v",
    ],
    verbose=True,
)

print("compile ok", INSTANCES)
