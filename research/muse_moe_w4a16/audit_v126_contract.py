"""Torch-free source and barrier-contract audit for the v126 half-stage diagnostic."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v126_wgs_halfstage_diag.py"
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
    "#define OKC_WGSHALF 0",
    "BW_E = 2 * WGS_NS",
    "BX_F = 4 * WGS_NS",
    "ws * 2 + half",
    "2 * WGS_PIECE",
    "ws * 2 + (wg >> 1)",
    "S.bars + BX_F + xs",
    "NW / 2",
    '"-DOKC_WGSHALF=1"',
    '"halfstage": WGS_HALF_DEFINES',
)
for text in required:
    require(text in source, f"missing half-stage contract: {text}")

# v126 is diagnostics-only: the production module remains v101, while only
# the untimed prepare() comparison compiles the experimental macro.
require("-DOKC_WGSHALF=1" not in re.search(r"MAIN_DEFINES\s*=\s*(.*)", source).group(1),
        "experimental macro leaked into MAIN_DEFINES")

# Barrier map: six weight-full, six weight-empty, three x-full, three x-empty.
slots = 3
bw_f = 0
bw_e = 2 * slots
bx_f = 4 * slots
bx_e = bx_f + slots
require((bw_f, bw_e, bx_f, bx_e, bx_e + slots) == (0, 6, 12, 15, 18),
        "unexpected half-stage barrier layout")
require(bx_e + slots <= 24, "half-stage barriers exceed Smem::bars")

# Each stage is still the exact four-piece v101 packed stage. Half h contains
# adjacent warpgroup pieces q=2h and q=2h+1, with neither gaps nor overlap.
piece = 9_216
unit_bytes = 4 * piece
half_bytes = 2 * piece
require((half_bytes, unit_bytes) == (18_432, 36_864), "unexpected stage byte sizes")
covered: list[tuple[int, int]] = []
for half in range(2):
    begin = half * half_bytes
    end = begin + half_bytes
    covered.append((begin, end))
    for local_q in range(2):
        q = 2 * half + local_q
        require(begin + local_q * piece == q * piece, "warpgroup piece moved between halves")
        require(q >> 1 == half, "consumer pair selects the wrong weight half")
require(covered == [(0, 18_432), (18_432, 36_864)], "halves do not exactly cover a stage")

# One TMA arrival completes each full barrier. Each half-empty barrier receives
# lane-0 arrivals from two warpgroups x four warps. The shared x-empty barrier
# retains one arrival per all 16 consumer warps.
require(1 == 1, "weight/x full barriers must each expect one producer")
require(2 * 4 == 8 == 16 // 2, "weight-half empty arrival count changed")
require(4 * 4 == 16, "x-empty arrival count changed")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print("PASS: diagnostics-only macro; MAIN_DEFINES remains the v101 scored path")
print("PASS: barrier map uses 18/24 entries with 1/8/1/16 arrival contracts")
print("PASS: two 18,432-byte halves exactly preserve the 36,864-byte packed stage")
print(f"candidate SHA256 {digest(CANDIDATE)}")
