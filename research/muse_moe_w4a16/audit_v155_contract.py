"""Static contract and schedule audit for the diagnostics-only v155 dual fixup."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from analyze_v154_split_fixup import inspect


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "submission_v139_wgs_kperm3.py"
CANDIDATE = ROOT / "submission_v155_wgs_dualfix_isolated_diag.py"
V154 = ROOT / "submission_v154_wgs_dualfix_diag.py"
ACTIVE = ROOT / "submission.py"

BASELINE_SHA256 = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"
ACTIVE_SHA256 = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment missing in {path.name}")


require(digest(BASELINE) == BASELINE_SHA256, "frozen v139 baseline changed")
require(digest(ACTIVE) == ACTIVE_SHA256, "active recovery submission.py changed")

source = CANDIDATE.read_text()
ast.parse(source)
require(source.startswith("# okc: problem=muse_moe_w4a16\n"), "OKC problem marker changed")
require(assignment(CANDIDATE, "MAIN_DEFINES") == assignment(BASELINE, "MAIN_DEFINES"),
        "production MAIN_DEFINES no longer match frozen v139")
require("#define OKC_WGSDUAL 0" in source, "dual fixup must default off")
require('"dualfix": tuple(MAIN_DEFINES) + ("-DOKC_WGSDUAL=1",)' in source,
        "diagnostic dualfix variant missing")
require("WGS_FIXEDGES = OKC_WGSDUAL ? 2 : 1" in source, "compile-time edge-buffer count missing")
require("pb * WGS_FIXEDGES + edge" in source, "phase/edge fixup isolation missing")
require("wgs_fix(p, PB, 0)" in source and "wgs_fix(p, PB, 1)" in source,
        "finisher does not gather both contributor edge classes")
require("(size_t)(slot * 4 + wg)" in source, "split shares are not isolated by boundary owner")
require("*reinterpret_cast<float4*>(fb + nt * 512) = make_float4" in source,
        "dualfix path does not use plain vector stores")
require("const int edge = s0 == 0 ? 0 : 1" in source, "stage-0 and later contributors are not isolated")
require("if (s0 == 0)" in source and "red.global.add.v4.f32" in source,
        "plain-first/atomic-later protocol missing")
require("*reinterpret_cast<float4*>(fb1 + nt * 512) = make_float4(0.f" in source,
        "later-contributor accumulator is not reset by the finisher")
require("WGS_FIXF = 4 * 160 * 4 * 4096" in source, "Python scratch does not reserve dual-edge maximum")
require("DIAG = True" in source, "v154 diagnostic payload is not enabled")
require("scratch_by_variant = {name: _make_scratch(dev) for name in variants}" in source,
        "diagnostic variants do not own isolated scratch allocations")
require('ext.okc_run(state["w"], scratch, x, idt, tw, out' in source,
        "diagnostic capture does not dispatch with its isolated scratch")
require('ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, 0, 32' not in source,
        "diagnostic capture still aliases the production scratch")

# v155 is a harness-only repair.  The CUDA candidate must remain byte-for-byte
# identical to the already-audited v154 payload.
require(assignment(CANDIDATE, "CUDA_SRC") == assignment(V154, "CUDA_SRC"),
        "v155 CUDA_SRC changed relative to frozen v154")

# Stress the exact v139 range arithmetic with every supported job/grid bound and
# alternating cheapest/most-expensive jobs.  inspect() proves exact coverage,
# collision-free (CTA, edge) slots, contiguous contributor enumeration, and a
# stage sum of 16 for each split unit.
cases = 0
for grid in range(1, 161):
    for nj in range(1, 137):
        counts = tuple(1 if j & 1 else 64 for j in range(nj))
        for pb in (0, 1):
            inspect(counts, pb, grid)
            cases += 1

print(f"PASS: v155 isolated-diagnostic dual fixup contract; {cases:,} schedules; CUDA_SRC unchanged; active and frozen v139 hashes unchanged")
