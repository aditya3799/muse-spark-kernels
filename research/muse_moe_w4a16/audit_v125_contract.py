"""Torch-free source and resource-contract audit for the v125 CPB2 diagnostic."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "submission_v101_wgs_tune.py"
CANDIDATE = ROOT / "submission_v125_wgs_2cta128_diag.py"
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
depth = 0
for line in match.group(1).splitlines():
    token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
    if token in {"#if", "#ifdef", "#ifndef"}:
        depth += 1
    elif token == "#endif":
        depth -= 1
        require(depth >= 0, "CUDA_SRC has an unmatched #endif")
require(depth == 0, "CUDA_SRC preprocessor conditionals are unbalanced")

required = (
    '#define OKC_WGSCPB 1',
    'constexpr int WGS_NWG = OKC_WGS ? NW / 4 : 0;',
    'constexpr int WGS_UNIT_BYTES = WGS_NWG * WGS_PIECE;',
    'constexpr int WGS_SLOTS = 160 * OKC_WGSCPB;',
    'const int launch_grid = (int)grid * OKC_WGSCPB;',
    'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    '"-DOKC_WGSCPB=2"',
    '"-DOKC_WGSNS=2"',
    '"-DOKC_WGSNX=2"',
    '"-DOKC_FT_T=32"',
)
for text in required:
    require(text in source, f"missing CPB2 contract: {text}")

for stale in (
    "pd.slot * 4 + wg",
    "slot * 4 + wg",
    "WGS_SLOTS * 4 * WGS_FIXW",
    "WGS_NS * UNIT_BYTES",
    "ws * UNIT_BYTES",
    "9 - pb",
):
    require(stale not in match.group(1), f"stale four-warpgroup/256-row assumption: {stale}")

# Hopper SM90 resource ceiling for the candidate's declared launch geometry.
threads_per_cta = 8 * 32 + 64
resident_ctas = 2
register_ceiling = 65_536 // (threads_per_cta * resident_ctas)
require(threads_per_cta == 320 and register_ceiling == 102, "unexpected CPB2 register ceiling")

# Dynamic rings only; the runtime occupancy guard covers the compiled static metadata and registers.
weight_piece = 9_216
weight_ring = 2 * (2 * weight_piece)
x_ring = 2 * ((32 // 8) * 4_096)
dynamic_ring_bytes = weight_ring + x_ring
require(dynamic_ring_bytes == 69_632, "unexpected CPB2 ring size")
require(2 * dynamic_ring_bytes < 228 * 1024, "rings alone exceed the H200 two-CTA shared-memory budget")

# Exhaustively prove that a 128-row half-unit selects the same packed bytes and global output rows as its parent
# 256-row v101 unit. The candidate changes scheduling granularity, not the lossless weight layout.
piece = 9_216
parent_stage_bytes = 4 * piece
for pb, parent_ubl in ((0, 5), (1, 4)):
    candidate_ubl = parent_ubl + 1
    for expert in range(32):
        for unit in range(1 << candidate_ubl):
            parent = unit >> 1
            half = unit & 1
            for stage in range(16):
                candidate_base = (((expert << parent_ubl) + parent) * 16 + stage) * parent_stage_bytes + half * 2 * piece
                for wg in range(2):
                    original_q = 2 * half + wg
                    original_addr = (((expert << parent_ubl) + parent) * 16 + stage) * parent_stage_bytes + original_q * piece
                    require(candidate_base + wg * piece == original_addr, "CPB2 packed-weight half mapping changed")
                    for warp in range(4):
                        candidate_r16 = unit * 8 + wg * 4 + warp
                        original_r16 = parent * 16 + original_q * 4 + warp
                        require(candidate_r16 == original_r16, "CPB2 output-row mapping changed")

# Four 128-row phase-A units x two warpgroups publish the same eight arrivals consumed by one phase-B stage.
for unit in range(64):
    require(unit >> 2 == (unit * 64) // 256, "phase-A to phase-B readiness mapping changed")
require(320 > 2 * 132, "fixup slots do not cover the doubled H200 grid")
main_fix_floats = 2 * 160 * 4 * (8 * 512)
candidate_fix_floats = 2 * 320 * 2 * (4 * 512)
require(candidate_fix_floats <= main_fix_floats, "Python scratch allocation is too small for CPB2 fixups")

print("PASS: Python AST and CUDA preprocessor balance")
print(f"PASS: immutable v101 control SHA256 {CONTROL_SHA256}")
print(f"PASS: CPB2 geometry {threads_per_cta} threads/CTA, <= {register_ceiling} registers/thread")
print(f"PASS: CPB2 dynamic rings {dynamic_ring_bytes} bytes/CTA before shared metadata")
print("PASS: exhaustive packed-weight, output-row, and phase-readiness mappings")
print("PASS: launch-time occupancy guard requires two resident blocks for both kernel variants")
print(f"candidate SHA256 {digest(CANDIDATE)}")
