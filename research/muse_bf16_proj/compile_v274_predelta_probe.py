"""Compile the exact v274 predecoded-delta pair kernel for SM90a."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission_plain_pairplanes_predelta_v274.py"


def main() -> None:
    submission = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SUBMISSION
    variable = sys.argv[2] if len(sys.argv) > 2 else "_V274_CUDA"
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc not found")
    tree = ast.parse(submission.read_text(encoding="utf-8"))
    cuda = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == variable
                   for t in node.targets):
                cuda = ast.literal_eval(node.value)
                break
    if not isinstance(cuda, str):
        raise RuntimeError("v274 CUDA marker mismatch")
    marker = "template <int NTH, int K>\nint64_t cgemv_pair_delta_info_t()"
    if cuda.count(marker) != 1:
        raise RuntimeError("v274 kernel marker mismatch")
    kernel = cuda.split(marker, 1)[0].replace(
        "cg_hib(", "probe_cg_hib(")
    source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ uint32_t probe_cg_hib(uint32_t c,
                                                  uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
''' + kernel + r'''

extern "C" void instantiate_v274(
    const uint8_t* p, const __nv_bfloat16* x, const int2* etab, int pmax,
    __nv_bfloat16* out, uint32_t base4, cudaStream_t stream) {
  cgemv_pair_delta_kernel<256,8192><<<1,288,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair_delta_kernel<512,8192><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair_delta_kernel<64,1024><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair_delta_kernel<32,1024><<<1,64,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair_delta_kernel<128,4096><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair_delta_kernel<256,4096><<<1,288,0,stream>>>(p,x,etab,pmax,out,base4);
}
'''
    with tempfile.TemporaryDirectory(prefix="v274_sm90a_") as tmp:
        src = Path(tmp) / "v274_probe.cu"
        obj = Path(tmp) / "v274_probe.obj"
        src.write_text(source, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [nvcc, "-std=c++17", "-O3", "-arch=sm_90a",
             "--ptxas-options=-v", "-c", str(src), "-o", str(obj)],
            text=True, capture_output=True, check=False)
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode:
            raise SystemExit(proc.returncode)
    print("V274 EXACT PREDELTA KERNEL SM90A COMPILE OK")


if __name__ == "__main__":
    main()
