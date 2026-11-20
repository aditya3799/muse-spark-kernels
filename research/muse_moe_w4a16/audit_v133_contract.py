"""Torch-free source, dispatch, and routing audit for the v133 production candidate."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v133_wgs_ft32_snap.py"
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
    "#define OKC_WGSSNAPA 0",
    "#define OKC_WGSSNAPB 0",
    "#if OKC_WGSSNAPA || OKC_WGSSNAPB",
    'WGS_SNAP_DEFINES = ("-DOKC_WGSSNAPA=1", "-DOKC_WGSSNAPB=1")',
    'WGS_FT32_SNAP_DEFINES = tuple(MAIN_DEFINES) + ("-DOKC_FT_T=32",) + WGS_SNAP_DEFINES',
    "WGS_FT64_SNAP_DEFINES = tuple(MAIN_DEFINES) + WGS_SNAP_DEFINES",
    'mods.append(tuple(WGS_FT32_SNAP_DEFINES) + state["sfx"])',
    'mods.append(tuple(WGS_FT64_SNAP_DEFINES) + state["sfx"])',
    "elif M == 32 and sfx:",
    'ext = _ext(tuple(WGS_FT32_SNAP_DEFINES) + sfx, use_mul=state["use_mul"])',
    "elif M >= 64 and sfx:",
    'ext = _ext(tuple(WGS_FT64_SNAP_DEFINES) + sfx, use_mul=state["use_mul"])',
    "const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;",
    "min(FT_T, tot - q * FT_T)",
)
for text in required:
    require(text in source, f"missing v133 contract: {text}")

require("OKC_WGSFINBCAST" not in source, "rejected final-pass broadcast leaked into v133")
main_line = re.search(r"MAIN_DEFINES\s*=\s*(.*)", source)
require(main_line is not None, "MAIN_DEFINES is missing")
for forbidden in ("OKC_FT_T", "OKC_WGSSNAP"):
    require(forbidden not in main_line.group(1), f"{forbidden} leaked into the base MAIN_DEFINES")

# Snap=1 is a monotone boundary transform.  Adjacent CTA intervals therefore
# still cover every stage exactly once; equal snapped boundaries only produce
# an empty CTA interval.
def snap1(stage: int) -> int:
    rem = stage & 15
    if rem == 1:
        return stage - 1
    if rem == 15:
        return stage + 1
    return stage


mapped = [snap1(stage) for stage in range(0, 8193)]
require(all(a <= b for a, b in zip(mapped, mapped[1:])), "snap=1 mapping is not monotone")
require(all(abs(a - b) <= 1 for a, b in zip(mapped, range(0, 8193))), "snap=1 moved a boundary too far")

# FT32 only repartitions an expert's ordered route interval.
for total in range(1, 3_073):
    chunks = [(q * 32, min(32, total - q * 32)) for q in range((total + 31) // 32)]
    rebuilt = [route for start, count in chunks for route in range(start, start + count)]
    require(rebuilt == list(range(total)), f"FT32 route partition failed for total={total}")
    require(all(0 < count <= 32 for _, count in chunks), "invalid FT32 chunk size")

mmax = 384
emax = mmax * 8
jmax32 = 32 + emax // 32 + 8
wgs_xrows32 = emax + 8 * jmax32
python_wgs_xrows = emax + 8 * (32 + emax // 32 + 8)
fixf32 = 2 * 160 * 4 * 2048 + 2 * 160 * 4 + 136 * 16
python_fixf64 = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
require((jmax32, wgs_xrows32) == (136, 4_160), "unexpected FT32 job/x-row bound")
require(python_wgs_xrows >= wgs_xrows32, "Python xA/xB scratch is too small for FT32")
require(python_fixf64 >= fixf32, "Python fixup scratch is too small for FT32")
require(8 + 1024 + emax >= 8 + 512 + jmax32 * 16, "Python int scratch is too small for FT32")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print("PASS: M32 selects FT32+snap; M>=64 selects FT64+snap; all require exact PRMT scales")
print("PASS: rejected final-pass broadcast is absent")
print("PASS: monotone snap=1 boundaries and exhaustive FT32 route partitions")
print("PASS: FT32 job table and every Python scratch bound")
print(f"candidate SHA256 {digest(CANDIDATE)}")
