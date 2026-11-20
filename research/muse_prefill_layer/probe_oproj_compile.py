"""Compile-check the embedded SM90a O-projection without importing submission.py."""

import ast
import pathlib

from torch.utils.cpp_extension import load_inline


path = pathlib.Path(__file__).with_name("submission.py")
tree = ast.parse(path.read_text())
keep = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_oproj_wgmma_fn":
        keep.append(node)
    elif isinstance(node, ast.Assign):
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & {"_OPROJ_CONFIGS", "_OPROJ_CUDA"}:
            keep.append(node)

namespace = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
configs = namespace["_OPROJ_CONFIGS"]
widths = sorted({config[0] for config in configs})
functions = "".join(namespace["_oproj_wgmma_fn"](width) for width in widths)
init_calls = " ".join(
    f"if constexpr (BN == {width}) wgmma{width}_init("
    "acc, smem_desc(x_addr), smem_desc(w_addr));"
    for width in widths
)
accum_calls = " ".join(
    f"if constexpr (BN == {width}) wgmma{width}_accum(acc, "
    "smem_desc(x_addr) + 2 * j, smem_desc(w_addr) + 2 * j);"
    for width in widths
)
cases = "\n".join(
    f"    case {index}: return launch_oproj<{bn}, {stages}, {nwg}>"
    "(p, x.data_ptr(), w.data_ptr(), stream);"
    for index, (bn, stages, nwg) in enumerate(configs)
)
source = (
    namespace["_OPROJ_CUDA"]
    .replace("OPROJ_WGMMA_FNS", functions)
    .replace("OPROJ_WGMMA_INIT", init_calls)
    .replace("OPROJ_WGMMA_ACCUM", accum_calls)
    .replace("OPROJ_CASES", cases)
)

load_inline(
    "muse_prefill_oproj_tma_compilecheck_v4",
    cpp_sources=(
        "int64_t oproj_run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor);"
    ),
    cuda_sources=source,
    functions=["oproj_run"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
    verbose=True,
)
print("SM90a extension compile OK")
