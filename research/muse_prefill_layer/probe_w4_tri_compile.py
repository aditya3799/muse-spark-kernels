import ast
import pathlib

from torch.utils.cpp_extension import load_inline


tree = ast.parse(pathlib.Path("submission.py").read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_w4g_wgm":
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "W4G_CUDA" for t in node.targets):
        nodes.append(node)
ns = {}
exec(compile(ast.Module(nodes, type_ignores=[]), "submission.py", "exec"), ns)

mod = load_inline(
    "pf2_w4g_pipe8_sm90_probe",
    cpp_sources=(
        "void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t); "
        "int64_t w4g_smem(); int64_t w4g_err(int64_t); int64_t w4g_ncl();"
    ),
    cuda_sources=ns["W4G_CUDA"],
    functions=["w4g", "w4g_smem", "w4g_err", "w4g_ncl"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "-lineinfo", "--ptxas-options=-v",
        "-gencode=arch=compute_90a,code=sm_90a", "-DHAS_WG_HOST=1", "-DDQ8=1", "-DSSPIPE=0",
    ],
    verbose=True,
)
print("SM90a compile/load OK", mod.w4g_smem())
