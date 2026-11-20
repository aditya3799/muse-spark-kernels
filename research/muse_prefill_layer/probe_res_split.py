import ast
import pathlib

import torch
from torch.utils.cpp_extension import load_inline


tree = ast.parse(pathlib.Path("submission.py").read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PF2_CUDA" for t in node.targets):
        nodes.append(node)
ns = {}
exec(compile(ast.Module(nodes, type_ignores=[]), "submission.py", "exec"), ns)

decl = """
void res_b(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void res_base(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void res_bp(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
"""
mod = load_inline(
    "pf2_res_split_probe_v2",
    cpp_sources=decl,
    cuda_sources=ns["PF2_CUDA"],
    functions=["res_b", "res_base", "res_bp"],
    extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
    verbose=True,
)


dev = "cuda"
T, H = 512, 8192
torch.manual_seed(47)
f = torch.randn((T, H), device=dev, dtype=torch.bfloat16)
a = torch.randn_like(f)
s = torch.randn((T, H), device=dev, dtype=torch.float32)
ra = (0.5 + torch.rand((T,), device=dev)).float()
coef = [torch.randn((H,), device=dev, dtype=torch.float32) for _ in range(6)]
al1, be1, al2, be2, gf, gn = coef
so0, so1 = torch.empty_like(s), torch.empty_like(s)
h0, h1 = torch.empty_like(f), torch.empty_like(f)
base = torch.empty_like(s)

mod.res_b(f, gf, a, ra, s, al1, be1, al2, be2, gn, so0, h0)
mod.res_base(a, ra, s, al1, be1, al2, base)
mod.res_bp(f, gf, base, be2, gn, so1, h1)
torch.cuda.synchronize()

sd = (so0 - so1).abs()
hd = (h0.float() - h1.float()).abs()
print(
    "parity",
    "stream_max", float(sd.max()),
    "stream_rel", float(sd.norm() / so0.norm()),
    "hidden_max", float(hd.max()),
    "hidden_rel", float(hd.norm() / h0.float().norm()),
    "hidden_equal", bool(torch.equal(h0, h1)),
)
assert float(sd.norm() / so0.norm()) < 1e-6
assert float(hd.norm() / h0.float().norm()) < 2e-5


def timing(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(11):
        beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
        beg.record()
        for _ in range(20):
            fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(beg.elapsed_time(end) * 50.0)
    vals.sort()
    return vals[len(vals) // 2]


t_ref = timing(lambda: mod.res_b(f, gf, a, ra, s, al1, be1, al2, be2, gn, so0, h0))
t_base = timing(lambda: mod.res_base(a, ra, s, al1, be1, al2, base))
t_tail = timing(lambda: mod.res_bp(f, gf, base, be2, gn, so1, h1))
print(f"local timing us: resb={t_ref:.1f} base={t_base:.1f} tail={t_tail:.1f} sequential={t_base + t_tail:.1f}")
