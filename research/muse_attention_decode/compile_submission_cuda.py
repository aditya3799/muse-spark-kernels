"""Compile the current submission's CUDA for real sm_90a and print ptxas resources."""

import ast
import hashlib
import os
import pathlib
import sys

from torch.utils.cpp_extension import load_inline


path = pathlib.Path("submission.py")
source = path.read_text()
tree = ast.parse(source)
cuda_source = None
debug_flags = []
source_config = {}

for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CUDA":
                cuda_source = ast.literal_eval(node.value)
            elif isinstance(target, ast.Name) and target.id == "_DBG":
                debug_flags = ast.literal_eval(node.value)
            elif isinstance(target, ast.Name) and target.id.startswith("_FD"):
                try:
                    source_config[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

if cuda_source is None:
    raise RuntimeError("current submission CUDA extension definition not found")

if "--baseline" in sys.argv:
    debug_flags = [flag for flag in debug_flags if flag != "-DAP_LDMT=1"]

if "--fdec-baseline" in sys.argv:
    debug_flags = [flag for flag in debug_flags if flag != "-DFD32_LDMT=1"]

if "--fdec" in sys.argv or "--fdec-baseline" in sys.argv:
    # Keep the exact device code through fdec but omit the production host dispatch.
    # Taking these specializations' addresses forces ptxas to compile the scored paths.
    cuda_source = cuda_source.split("void prep_only", 1)[0]
    cuda_source += r"""
void resource_probe(int64_t smem) {
  cudaFuncSetAttribute(fdec<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute(fdec<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
}
"""
else:
    # Keep the exact device code through attnp but omit later kernels and the large
    # production dispatch table. Taking this specialization's address forces ptxas
    # to compile precisely the scored full-verify path.
    cuda_source = cuda_source.split("static int g_cmrec", 1)[0]
    cuda_source += r"""
__device__ __forceinline__ void rope_cs2(float invf, int pos, float& sn, float& cs) {
  const float ang = (float)pos * invf;
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs) {
  const float ang = (float)pos * invf[pair];
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}
void resource_probe(int64_t smem) {
  cudaFuncSetAttribute(attnp<2, 537002308>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
}
"""

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
tag = hashlib.sha256((source + repr(debug_flags)).encode()).hexdigest()[:12]
fdec_rbf = 7 if "-DFD_REGPT=1" in debug_flags else (6 if source_config.get("_FD32_RBF6") else 5)
load_inline(
    name=f"okc_attention_sm90_resources_{tag}",
    cpp_sources="void resource_probe(int64_t);",
    cuda_sources=cuda_source,
    functions=["resource_probe"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-DRB=8",
        "-DRB2=6",
        "-DRB3=20",
        f"-DRBF={fdec_rbf}",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--ptxas-options=-v",
    ] + debug_flags,
    with_cuda=True,
    verbose=True,
)
print("sm_90a CUDA extension compile OK")
