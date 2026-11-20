"""Torch-free source, partition, and scratch audit for the v131 WGS FT sweep."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v131_wgs_ft_sweep.py"
CONTROL_SHA256 = "A88B923A04B8DE604DCE5C82EFE607F8DD43D2CD1E753A75B7B5D633B52FAFFD"
WIDTHS = (40, 48, 56)


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

for width in WIDTHS:
    require(
        f'WGS_FT{width}_DEFINES = tuple(MAIN_DEFINES) + ("-DOKC_FT_T={width}",)' in source,
        f"missing FT{width} diagnostic defines",
    )
    require(f'"ft{width}": WGS_FT{width}_DEFINES' in source, f"missing FT{width} diagnostic variant")

required = (
    "const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;",
    "min(FT_T, tot - q * FT_T)",
    "constexpr int WGS_NT = FT_T / 8;",
    "#if OKC_FT_T >= 40",
    "#if OKC_FT_T >= 48",
    "#if OKC_FT_T >= 56",
    "#if OKC_FT_T >= 64",
)
for text in required:
    require(text in source, f"missing FT sweep contract: {text}")
require("#if OKC_FT_T > 32" not in cuda, "broad FT guard would instantiate out-of-range templates")

main_line = re.search(r"MAIN_DEFINES\s*=\s*(.*)", source)
require(main_line is not None and "OKC_FT_T" not in main_line.group(1),
        "FT sweep leaked into the scored MAIN_DEFINES path")

# At FT64 every new guard is true, so the scored control retains the original
# case 5/6/7/default-8 sequence.  Narrower builds instantiate only reachable
# specializations, avoiding the invalid NTT > WGS_NT register allocation.
case_positions = [cuda.index(f"case {n}: wgs_seg<PB, {n}>") for n in (5, 6, 7)]
case_positions.append(cuda.index("default: wgs_seg<PB, 8>"))
require(case_positions == sorted(case_positions), "FT64 switch order changed")

# Every width only repartitions each expert's contiguous route interval.  Prove
# coverage, order, and chunk limits for every possible local route total.
for width in WIDTHS:
    for total in range(1, 3_073):
        chunks = [(q * width, min(width, total - q * width))
                  for q in range((total + width - 1) // width)]
        rebuilt = [route for start, count in chunks for route in range(start, start + count)]
        require(rebuilt == list(range(total)), f"FT{width} route partition failed for total={total}")
        require(all(0 < count <= width for _, count in chunks), f"invalid FT{width} chunk size")

mmax = 384
emax = mmax * 8
python_wgs_xrows = emax + 8 * (32 + emax // 32 + 8)
python_fixf64 = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
python_ints = 8 + 1024 + emax
for width in WIDTHS:
    jmax = 32 + emax // width + 8
    wgs_xrows = emax + 8 * jmax
    fixw = (width // 8) * 512
    fixf = 2 * 160 * 4 * fixw + 2 * 160 * 4 + 136 * 16
    require(python_wgs_xrows >= wgs_xrows, f"Python xA/xB scratch is too small for FT{width}")
    require(python_fixf64 >= fixf, f"Python fixup scratch is too small for FT{width}")
    require(python_ints >= 8 + 512 + jmax * 16, f"Python int scratch is too small for FT{width}")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print("PASS: FT40/48/56 are diagnostics-only; scored FT64 switch is unchanged")
print("PASS: exhaustive contiguous route partitions for totals 1..3072")
print("PASS: FT40/48/56 job tables and all Python scratch bounds")
print(f"candidate SHA256 {digest(CANDIDATE)}")
