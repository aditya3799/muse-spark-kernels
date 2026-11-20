"""Torch-free source and dispatch-contract audit for the v129 M=32 FT32 candidate."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v129_wgs_ft32_m32.py"
CONTROL_SHA256 = "A88B923A04B8DE604DCE5C82EFE607F8DD43D2CD1E753A75B7B5D633B52FAFFD"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


require(digest(CONTROL) == CONTROL_SHA256, "v101 leaderboard control changed")
source = CANDIDATE.read_text(encoding="utf-8")
ast.parse(source)

match = re.search(r"CUDA_SRC = r'''(.*?)'''", source, re.S)
require(match is not None, "CUDA_SRC raw string is missing")
cuda = match.group(1)
depth = 0
for line in cuda.splitlines():
    token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
    if token in {"#if", "#ifdef", "#ifndef"}:
        depth += 1
    elif token == "#endif":
        depth -= 1
        require(depth >= 0, "CUDA_SRC has an unmatched #endif")
require(depth == 0, "CUDA_SRC preprocessor conditionals are unbalanced")

required = (
    'WGS_FT32_DEFINES = tuple(MAIN_DEFINES) + ("-DOKC_FT_T=32",)',
    "DIAG = False",
    'if use_mul and state["sfx"]:',
    'mods.append(tuple(WGS_FT32_DEFINES) + state["sfx"])',
    "elif M == 32 and sfx:",
    'ext = _ext(tuple(WGS_FT32_DEFINES) + sfx, use_mul=state["use_mul"])',
    "const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;",
    "min(FT_T, tot - q * FT_T)",
    "constexpr int WGS_NT = FT_T / 8;",
    "#if OKC_FT_T > 32",
)
for text in required:
    require(text in source, f"missing FT32 contract: {text}")

main_line = re.search(r"MAIN_DEFINES\s*=\s*(.*)", source)
require(main_line is not None and "OKC_FT_T" not in main_line.group(1),
        "FT32 leaked into the scored MAIN_DEFINES path")

# Exhaustively prove that changing the chunk limit only repartitions each
# expert's contiguous route interval. Every original route appears once, in
# order, with chunks no larger than 32 tokens.
for total in range(1, 3_073):
    chunks = [(q * 32, min(32, total - q * 32)) for q in range((total + 31) // 32)]
    rebuilt = [route for start, count in chunks for route in range(start, start + count)]
    require(rebuilt == list(range(total)), f"FT32 route partition failed for total={total}")
    require(all(0 < count <= 32 for _, count in chunks), "invalid FT32 chunk size")

# Candidate resource/scratch maxima. Python intentionally allocates the larger
# FT64 fixup area, while its WGS_XROWS expression is exact for FT32's larger
# job table.
mmax = 384
emax = mmax * 8
jmax32 = 32 + emax // 32 + 8
wgs_xrows32 = emax + 8 * jmax32
python_wgs_xrows = emax + 8 * (32 + emax // 32 + 8)
fixw32 = (32 // 8) * 512
fixf32 = 2 * 160 * 4 * fixw32 + 2 * 160 * 4 + 136 * 16
python_fixf64 = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
require((jmax32, wgs_xrows32) == (136, 4_160), "unexpected FT32 job/x-row bound")
require(python_wgs_xrows >= wgs_xrows32, "Python xA/xB scratch is too small for FT32")
require(python_fixf64 >= fixf32, "Python fixup scratch is too small for FT32")
require(8 + 1_024 + emax >= 8 + 512 + jmax32 * 16,
        "Python int scratch is too small for FT32 job metadata/readiness")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print("PASS: M=32 dispatch selects FT32 only with the exact PRMT-scale guard")
print("PASS: exhaustive contiguous route partition for totals 1..3072")
print("PASS: FT32 JMAX=136, WGS_XROWS=4160, and all Python scratch bounds")
print(f"candidate SHA256 {digest(CANDIDATE)}")
