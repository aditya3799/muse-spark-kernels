"""Torch-free source and numerical contract audit for v77."""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if not (ROOT / "submission_v77.py").exists():
    ROOT = Path(r"D:\OpenKernelChallenge\muse_moe_w4a16")

V64_SHA256 = "F0AD063A8A89FC3E40437E8C5BA946C675BFF1BFC0E3866BC1261B50661DB4E6"
V77_SHA256 = "71A79837CC7ED8C8E6A7CBF0F650A0FF96583F15AA7F509ECA346740D32CD19B"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


candidate = ROOT / "submission_v77.py"
source = candidate.read_text(encoding="utf-8")
ast.parse(source)

for name in ("M_CUDA_K", "M_CUDA_X"):
    match = re.search(name + r' = r"""(.*?)"""', source, re.S)
    require(match is not None, f"{name} raw string is missing")
    depth = 0
    for line in match.group(1).splitlines():
        token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if token in {"#if", "#ifdef", "#ifndef"}:
            depth += 1
        elif token == "#endif":
            depth -= 1
            require(depth >= 0, f"{name} has an unmatched #endif")
    require(depth == 0, f"{name} preprocessor conditionals are unbalanced")

require(digest(ROOT / "submission.py") == V64_SHA256, "active submission.py is not v64")
require(digest(ROOT / "submission_v64.py") == V64_SHA256, "v64 control hash changed")
require(digest(candidate) == V77_SHA256, "v77 candidate hash changed")

require(source.count("#ifdef POLYSILU") == 2, "POLYSILU must exist only in gk and wx")
require(source.count("return x / (1.f + __expf(-x));") == 2, "exact SiLU fallbacks are missing")
require(source.count("2.82057459e-4f") == 2, "polynomial coefficients changed")
require(source.count("silu_eval(gt)") == 3, "not every selected gk/wx SiLU site uses the helper")
require("#define POLYSILU" not in source, "POLYSILU must default off")

for cfg in (
    '"-DSW=4 -DKS2=1 -DMAXM=8 -DPOLYSILU"',
    '"-DSW=1 -DKS2=2 -DPOLYSILU"',
    '"-DSW=1 -DKS1=2 -DKS2=4 -DMAXM=32 -DPOLYSILU"',
    '"-DSW=1 -DKS2=2 -DMAXM=8 -DPOLYSILU"',
):
    require(source.count(cfg) == 1, f"missing focused gk candidate {cfg}")

require(source.count('("-DNCW=16", "-DWGPAIR", "-DEPISHADOW=1", "-DPOLYSILU"),') == 1,
        "combined M256 shadow/polynomial candidate is missing")
require(source.count('("-DNCW=8", "-DRPW=2", "-DTPX=32", "-DNSL=2", "-DCPB=2", "-DPOLYSILU"),') == 1,
        "M128 CPB2 polynomial candidate is missing")
require('("-DNCW=16", "-DWGPAIR"),' in source, "v64 M256 control is missing")
require('("-DNCW=8", "-DRPW=2", "-DTPX=32", "-DNSL=2", "-DCPB=2"),' in source,
        "v64 M128 control is missing")
require("setmaxnreg" not in source and "reg_inc" not in source, "rejected dynamic register allocation returned")


def exact_silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def poly_silu(x: float) -> float:
    z = x * x
    p = (2.82057459e-4 * z - 1.16906086e-2) * z + 2.34261343e-1
    return x * min(1.0, max(0.0, 0.5 + x * p))


rng = random.Random(5252)
for std, limit in ((1.0, 0.011), (3.2, 0.008)):
    err2 = ref2 = 0.0
    for _ in range(200_000):
        x = rng.gauss(0.0, std)
        ref = exact_silu(x)
        err2 += (poly_silu(x) - ref) ** 2
        ref2 += ref * ref
    require(math.sqrt(err2 / ref2) < limit, f"polynomial SiLU exceeds the std={std} scalar error bound")

max_abs = max(abs(poly_silu(-10.0 + i / 10_000.0) - exact_silu(-10.0 + i / 10_000.0))
              for i in range(200_001))
require(max_abs < 0.054, "polynomial SiLU uniform-range maximum error changed")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: active/control v64 SHA256 {V64_SHA256}")
print("PASS: exact SiLU fallbacks and all six focused POLYSILU candidates retained")
print("PASS: sampled normal/skewed scalar error and uniform max-error bounds")
print("PASS: rejected setmaxnreg mechanism remains absent")
print(f"candidate SHA256 {digest(candidate)}")
