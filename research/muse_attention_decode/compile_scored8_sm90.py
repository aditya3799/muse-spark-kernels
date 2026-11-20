"""Compile a submission's transformed scored extension without importing SGLang."""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import sys

from torch.utils.cpp_extension import load_inline


if len(sys.argv) != 2:
    raise SystemExit("usage: compile_scored8_sm90.py SUBMISSION.py")

path = pathlib.Path(sys.argv[1]).resolve()
tree = ast.parse(path.read_text(encoding="utf-8"))
values: dict[str, str] = {}
transform: dict[str, str] = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"CUDA_SRC", "CPP_SRC"}:
                values[target.id] = ast.literal_eval(node.value)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_slim_cuda_source":
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id in {"old", "new"}:
                    transform[target.id] = ast.literal_eval(statement.value)

cuda_source = values["CUDA_SRC"]
cpp_source = values["CPP_SRC"]
old, new = transform["old"], transform["new"]
if cuda_source.count(old) != 1:
    raise RuntimeError("scored launch source anchor changed")
cuda_source = cuda_source.replace(old, new, 1)

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
tag = hashlib.sha256(cuda_source.encode()).hexdigest()[:12]
load_inline(
    name=f"okc_attention_scored8_{tag}",
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
print(f"scored-8 sm_90a extension compile OK: {path.name}")
