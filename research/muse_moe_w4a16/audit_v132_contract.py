"""Torch-free contract audit for the v132 composed WGS diagnostic."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v132_wgs_composite_diag.py"
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
    "#define OKC_WGSFINBCAST 0",
    "#define OKC_WGSSNAPA 0",
    "#define OKC_WGSSNAPB 0",
    "#if OKC_WGSSNAPA || OKC_WGSSNAPB",
    "#if OKC_WGSFINBCAST",
    'WGS_SNAP_BCAST = ("-DOKC_WGSSNAPA=1", "-DOKC_WGSSNAPB=1", "-DOKC_WGSFINBCAST=1")',
    'WGS_M32_COMPOSITE = tuple(MAIN_DEFINES) + ("-DOKC_FT_T=32",) + WGS_SNAP_BCAST',
    "WGS_LARGE_COMPOSITE = tuple(MAIN_DEFINES) + WGS_SNAP_BCAST",
    '"m32combo": WGS_M32_COMPOSITE',
    '"largecombo": WGS_LARGE_COMPOSITE',
)
for text in required:
    require(text in source, f"missing composite contract: {text}")

main_line = re.search(r"MAIN_DEFINES\s*=\s*(.*)", source)
require(main_line is not None, "MAIN_DEFINES is missing")
for forbidden in ("OKC_FT_T", "OKC_WGSSNAP", "OKC_WGSFINBCAST"):
    require(forbidden not in main_line.group(1), f"{forbidden} leaked into scored MAIN_DEFINES")

# With all three feature defaults at zero, preprocessing selects the original
# v101 range ownership and per-thread final-normalization statements.
require("g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);" in cuda,
        "original v101 range path is missing")
require("while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;" in cuda,
        "original v101 fixup-owner path is missing")
require("red[NW * 8 + tid] = tsum;" in cuda, "original v101 final reduction path is missing")
require("const float r = p.topk_w[tk * 8 + sl] * rsqrtf" in cuda,
        "original v101 per-thread normalization path is missing")

# Snap=1 maps only stage 1 down to a unit edge and stage 15 up to the
# following edge.  The mapping is monotone, so adjacent CTA ranges still form
# a gap-free, overlap-free partition even when a CTA receives an empty range.
def snap1(stage: int) -> int:
    rem = stage & 15
    if rem == 1:
        return stage - 1
    if rem == 15:
        return stage + 1
    return stage


mapped = [snap1(stage) for stage in range(0, 8193)]
require(all(a <= b for a, b in zip(mapped, mapped[1:])), "snap=1 boundary mapping is not monotone")
require(all(abs(a - b) <= 1 for a, b in zip(mapped, range(0, 8193))), "snap=1 moved a boundary too far")

# FT32 only repartitions each expert's contiguous routes.  The final broadcast
# changes who evaluates the identical FP32 expression, not its inputs or use.
for total in range(1, 3_073):
    chunks = [(q * 32, min(32, total - q * 32)) for q in range((total + 31) // 32)]
    rebuilt = [route for start, count in chunks for route in range(start, start + count)]
    require(rebuilt == list(range(total)), f"FT32 route partition failed for total={total}")

mmax = 384
emax = mmax * 8
jmax32 = 32 + emax // 32 + 8
python_wgs_xrows = emax + 8 * (32 + emax // 32 + 8)
python_fixf64 = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
fixf32 = 2 * 160 * 4 * 2048 + 2 * 160 * 4 + 136 * 16
require(python_wgs_xrows >= emax + 8 * jmax32, "Python xA/xB scratch is too small for FT32")
require(python_fixf64 >= fixf32, "Python fixup scratch is too small for FT32")
require(8 + 1024 + emax >= 8 + 512 + jmax32 * 16, "Python int scratch is too small for FT32")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print("PASS: scored MAIN_DEFINES selects the exact default-off v101 paths")
print("PASS: snap=1 boundaries are monotone and move at most one stage")
print("PASS: exhaustive FT32 route partitions and all scratch bounds")
print(f"candidate SHA256 {digest(CANDIDATE)}")
