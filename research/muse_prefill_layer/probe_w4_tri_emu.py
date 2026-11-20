import ast
import pathlib

import torch
from nvfp4 import random_weights
from torch.utils.cpp_extension import load_inline


tree = ast.parse(pathlib.Path("submission.py").read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_w4g_wgm":
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in ("Q__place", "_w4pack"):
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "W4G_CUDA" for t in node.targets):
        nodes.append(node)
ns = {"torch": torch}
exec(compile(ast.Module(nodes, type_ignores=[]), "submission.py", "exec"), ns)


def q_place_dq8(word, code, pair, hi):
    rotations = ((0, 0, 0, 0), (10, 10, 10, 10), (13, 13, 13, 13), (2, 7, 7, 12))
    for bit, tgt, delta in zip(range(4), (6, 7, 8, 15), rotations[pair]):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - delta) % 32)
    return word


ns["Q__place"] = q_place_dq8

mod = load_inline(
    "pf2_w4g_pipe8_emu_probe",
    cpp_sources=(
        "void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t); "
        "int64_t w4g_smem(); int64_t w4g_err(int64_t); int64_t w4g_ncl();"
    ),
    cuda_sources=ns["W4G_CUDA"],
    functions=["w4g", "w4g_smem", "w4g_err", "w4g_ncl"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "-lineinfo", "-DBNMAX=64", "-DEMU_NS=4",
        "-DHAS_WG_HOST=0", "-DDQ8=1", "-DSSPIPE=0",
    ],
    verbose=True,
)


def check(rows, swiglu, ksplit):
    # Small but layout-complete matrices for both epilogues.
    raw_rows = 512 if swiglu else 256
    gen = torch.Generator(device="cuda").manual_seed(17 + rows + swiglu)
    raw, sf, gs = random_weights(raw_rows, 4096, leading=(1,), generator=gen, device="cuda")
    w4, scale = ns["_w4pack"](raw, sf, gs, swiglu=swiglu)
    torch.manual_seed(31 + rows)
    a = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
    offs = torch.full((32,), rows, dtype=torch.int32, device="cuda")
    ctr0 = torch.zeros(8, dtype=torch.int32, device="cuda")
    ctr4 = torch.zeros(8, dtype=torch.int32, device="cuda")
    ctr8 = torch.zeros(8, dtype=torch.int32, device="cuda")
    prof = torch.empty(0, dtype=torch.int64, device="cuda")
    nout = raw_rows // 2 if swiglu else raw_rows
    out0 = torch.empty((ksplit * rows, nout), dtype=torch.bfloat16, device="cuda")
    out4 = torch.empty_like(out0)
    out8 = torch.empty_like(out0)
    mod.w4g(a, offs, w4, scale, out0, ctr0, prof, ksplit, 3, 1, int(swiglu), 0)
    mod.w4g(a, offs, w4, scale, out4, ctr4, prof, ksplit, 3, 5, int(swiglu), 0)
    mod.w4g(a, offs, w4, scale, out8, ctr8, prof, ksplit, 3, 6, int(swiglu), 0)
    torch.cuda.synchronize()
    assert mod.w4g_err(1) == 0
    af, b4, b8 = out0.float(), out4.float(), out8.float()
    rel4 = float((af - b4).norm() / af.norm())
    rel8 = float((af - b8).norm() / af.norm())
    mx8 = float((af - b8).abs().max())
    print(f"rows={rows} swiglu={swiglu} ksplit={ksplit}: rel4={rel4:.3e} rel8={rel8:.3e} max8={mx8:.3e} "
          f"norm0={float(af.norm()):.3e} norm8={float(b8.norm()):.3e} "
          f"nz0={int(torch.count_nonzero(out0))} nz8={int(torch.count_nonzero(out8))}")
    assert rel4 < 2e-3 and rel8 < 2e-3
    if rows == 384 and not swiglu and ksplit == 1:
        def timing(mode, out, ctr):
            for _ in range(8):
                mod.w4g(a, offs, w4, scale, out, ctr, prof, 1, 3, mode, 0, 0)
            torch.cuda.synchronize()
            vals = []
            for _ in range(9):
                beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
                beg.record()
                for _ in range(20):
                    mod.w4g(a, offs, w4, scale, out, ctr, prof, 1, 3, mode, 0, 0)
                end.record(); torch.cuda.synchronize(); vals.append(beg.elapsed_time(end) * 50.0)
            return sorted(vals)[len(vals) // 2]
        t0 = timing(1, out0, ctr0)
        t4 = timing(5, out4, ctr4)
        t8 = timing(6, out8, ctr8)
        print(f"paired local timing: wide={t0:.1f} us pipe4={t4:.1f} us pipe8={t8:.1f} us "
              f"ratios={t4 / t0:.3f}/{t8 / t0:.3f}")


print("GPU", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "smem", mod.w4g_smem())
for args in ((384, True, 1), (211, True, 1), (384, False, 1), (211, False, 2), (19, False, 1)):
    check(*args)
print("pipe4/pipe8 emulation parity OK")
