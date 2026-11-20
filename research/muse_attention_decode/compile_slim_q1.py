"""Compile the slim q1-flat diagnostic extension for sm_90a without SGLang installed."""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import sys

from torch.utils.cpp_extension import load_inline


if len(sys.argv) != 2:
    raise SystemExit("usage: compile_slim_q1.py SUBMISSION.py")

path = pathlib.Path(sys.argv[1]).resolve()
tree = ast.parse(path.read_text(encoding="utf-8"))
values = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"CUDA_SRC", "CPP_SRC"}:
                values[target.id] = ast.literal_eval(node.value)

cuda_source = values["CUDA_SRC"]
cpp_source = values["CPP_SRC"]
old = "  if (L == 1) launch<1>(p, grid, stream, (int)cs, true, pdl);\n  else launch<8>(p, grid, stream, (int)cs, true, pdl);"
new = r'''  TORCH_CHECK(L == 1 && flat == 1, "q1-occ3 slim diagnostic only supports flat q_len=1");
  p.pdlf = 0;
  p.coop = 0;
  p.resident = 0;
  p.unev = 0;
  const int sm = (int)sizeof(Smem<1, true>) + 1024;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(attn_kernel<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    init = true;
  }
  attn_kernel<1, true><<<grid, Cfg<1, true>::NTX, sm, stream>>>(p);'''
if cuda_source.count(old) != 1:
    raise RuntimeError("slim launch source anchor changed")
cuda_source = cuda_source.replace(old, new, 1)

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
tag = hashlib.sha256(cuda_source.encode()).hexdigest()[:12]
load_inline(
    name=f"okc_attention_q1occ3_slim_{tag}",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["run"],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",
    ],
    with_cuda=True,
    verbose=True,
)
print(f"slim sm_90a extension compile OK: {path.name}")
