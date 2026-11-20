"""Compile the exact v272 grouped pair kernel for SM90a without importing torch."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission_plain_pairplanes_rowgroup_v272.py"


def literal_assignment(name: str) -> str:
    submission = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SUBMISSION
    tree = ast.parse(submission.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError(name)
                return value
    raise KeyError(name)


def main() -> None:
    variable = sys.argv[2] if len(sys.argv) > 2 else "_V272_CUDA"
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc not found")
    cuda = literal_assignment(variable)
    marker = "template <int NTH, int K, int GROUP>\nint64_t cgemv_pair_group_info_t()"
    if cuda.count(marker) != 1:
        raise RuntimeError("v272 kernel marker mismatch")
    kernel = cuda.split(marker, 1)[0]
    kernel = kernel.replace("cg_hib(", "probe_cg_hib(")
    kernel = kernel.replace("esc_delta(", "probe_esc_delta(")
    rolled = kernel.replace(
        "cgemv_pair_group_kernel", "cgemv_pair_group_rolled_kernel")
    rolled = rolled.replace(
        "#pragma unroll\n  for (int gg = 0; gg < GROUP; ++gg)",
        "#pragma unroll 1\n  for (int gg = 0; gg < GROUP; ++gg)")
    source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float probe_esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) -
         __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}

__device__ __forceinline__ uint32_t probe_cg_hib(uint32_t c,
                                                  uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
''' + kernel + rolled + r'''

extern "C" void instantiate_v272(
    const uint8_t* p, const __nv_bfloat16* x, const int2* etab, int pmax,
    __nv_bfloat16* out, uint32_t base4, int npairs, cudaStream_t stream) {
  cgemv_pair_group_kernel<512,8192,1><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<512,8192,2><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<512,8192,4><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<64,1024,1><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<64,1024,2><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<64,1024,4><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<128,4096,1><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<128,4096,2><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_kernel<128,4096,4><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<512,8192,2><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<512,8192,4><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<512,8192,8><<<1,544,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<64,1024,2><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<64,1024,4><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<64,1024,8><<<1,96,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<128,4096,2><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<128,4096,4><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
  cgemv_pair_group_rolled_kernel<128,4096,8><<<1,160,0,stream>>>(p,x,etab,pmax,out,base4,npairs);
}
'''
    with tempfile.TemporaryDirectory(prefix="v272_sm90a_") as tmp:
        src = Path(tmp) / "v272_probe.cu"
        obj = Path(tmp) / "v272_probe.obj"
        src.write_text(source, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [nvcc, "-std=c++17", "-O3", "-arch=sm_90a",
             "--ptxas-options=-v", "-c", str(src), "-o", str(obj)],
            text=True, capture_output=True, check=False)
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode:
            raise SystemExit(proc.returncode)
    print("V272 EXACT GROUP KERNEL SM90A COMPILE OK")


if __name__ == "__main__":
    main()
