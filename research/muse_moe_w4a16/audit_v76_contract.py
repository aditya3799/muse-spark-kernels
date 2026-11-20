"""Torch-free source contract audit for the unsubmitted v76 candidate."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if not (ROOT / "submission_v76.py").exists():
    ROOT = Path(r"D:\OpenKernelChallenge\muse_moe_w4a16")

V64_SHA256 = "F0AD063A8A89FC3E40437E8C5BA946C675BFF1BFC0E3866BC1261B50661DB4E6"
V76_SHA256 = "69CF44B8D8F801337EAFF876BD7E45A105CDE91332A8180259044BA7E67A7E90"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


candidate = ROOT / "submission_v76.py"
source = candidate.read_text(encoding="utf-8")
ast.parse(source)

match = re.search(r'M_CUDA_X = r"""(.*?)"""', source, re.S)
require(match is not None, "M_CUDA_X raw string is missing")
cuda = match.group(1)

depth = 0
for line in cuda.splitlines():
    token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
    if token in {"#if", "#ifdef", "#ifndef"}:
        depth += 1
    elif token == "#endif":
        depth -= 1
        require(depth >= 0, "unbalanced CUDA preprocessor close")
require(depth == 0, "unbalanced CUDA preprocessor conditionals")

require(digest(ROOT / "submission.py") == V64_SHA256, "active submission.py is not v64")
require(digest(ROOT / "submission_v64.py") == V64_SHA256, "v64 control hash changed")
require(digest(candidate) == V76_SHA256, "v76 candidate hash changed")

require("#define EPISHADOW 0" in cuda, "EPISHADOW must default off")
require("volatile unsigned long long g1p, g2p, nwp;" in cuda, "shared pointer shadow is missing")
require("sm->g1p = reinterpret_cast<unsigned long long>(p.g1);" in cuda, "g1 shadow initialization is missing")
require("sm->g2p = reinterpret_cast<unsigned long long>(p.g2);" in cuda, "g2 shadow initialization is missing")
require("sm->nwp = reinterpret_cast<unsigned long long>(p.nw);" in cuda, "nw shadow initialization is missing")
require(cuda.index("sm->nwp =") < cuda.index("__syncthreads();", cuda.index("__global__ void __launch_bounds__(NTH, CPB) moe_x")),
        "shared pointer shadow must be initialized before the first CTA barrier")
require("reinterpret_cast<const float*>(sm->g1p)" in cuda, "W1 epilogue does not reload g1 from shared")
require("reinterpret_cast<const float*>(sm->g2p)" in cuda, "W2 epilogue does not reload g2 from shared")
require("reinterpret_cast<const float*>(sm->nwp)" in cuda, "W2 epilogue does not reload nw from shared")
require("setmaxnreg" not in cuda and "reg_inc" not in cuda, "v76 must not use dynamic register allocation")
require("#define NTH ((NCW + 1) * 32)" in cuda, "CTA geometry changed")
require('("-DNCW=16", "-DWGPAIR"),' in source, "v64 WGPAIR control is missing")
candidate_tuple = '("-DNCW=16", "-DWGPAIR", "-DEPISHADOW=1"),'
require(source.count(candidate_tuple) == 1, "focused v76 tuner candidate must appear exactly once")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: active/control v64 SHA256 {V64_SHA256}")
print("PASS: EPISHADOW defaults off and shadows g1/g2/nw before the first CTA barrier")
print("PASS: original CTA geometry and v64 WGPAIR tuner control retained")
print("PASS: no setmaxnreg instruction source remains in v76")
print(f"candidate SHA256 {digest(candidate)}")
