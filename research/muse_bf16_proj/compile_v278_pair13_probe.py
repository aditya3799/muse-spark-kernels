"""Compile the exact v278 C13 pair kernel for SM90a."""

import ast
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission_plain_pairplanes_c13_nohelper_v278.py"


def main() -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc not found")
    tree = ast.parse(SUBMISSION.read_text(encoding="utf-8"))
    cuda = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_V278_CUDA"
                   for t in node.targets):
                cuda = ast.literal_eval(node.value)
                break
    if not isinstance(cuda, str):
        raise RuntimeError("v278 CUDA marker mismatch")
    marker = "template <int NTH, int K>\nint64_t cgemv_pair13_info_t()"
    if cuda.count(marker) != 1:
        raise RuntimeError("v278 kernel marker mismatch")
    kernel = cuda.split(marker, 1)[0]
    source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
''' + kernel + r'''

extern "C" void instantiate_v278(
    const uint8_t* p, const __nv_bfloat16* x, const int2* etab, int pmax,
    __nv_bfloat16* out, uint32_t base4, cudaStream_t stream) {
  cgemv_pair13_kernel<512,8192><<<1,512,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair13_kernel<64,1024><<<1,64,0,stream>>>(p,x,etab,pmax,out,base4);
  cgemv_pair13_kernel<128,4096><<<1,128,0,stream>>>(p,x,etab,pmax,out,base4);
}
'''
    with tempfile.TemporaryDirectory(prefix="v278_sm90a_") as tmp:
        src = Path(tmp) / "v278_probe.cu"
        obj = Path(tmp) / "v278_probe.obj"
        src.write_text(source, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [nvcc, "-std=c++17", "-O3", "-arch=sm_90a",
             "--ptxas-options=-v", "-c", str(src), "-o", str(obj)],
            text=True, capture_output=True, check=False)
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode:
            raise SystemExit(proc.returncode)
    print("V278 EXACT C13 PAIR KERNEL SM90A COMPILE OK")


if __name__ == "__main__":
    main()
