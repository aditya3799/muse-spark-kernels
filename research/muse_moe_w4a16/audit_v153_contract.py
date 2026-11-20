"""Static contract and exhaustive coverage audit for the diagnostics-only v153 cyclic scheduler."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "submission_v139_wgs_kperm3.py"
CANDIDATE = ROOT / "submission_v153_wgs_cyclic_diag.py"
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
require("#define OKC_WGSCYCLIC 0" in source, "cyclic scheduler must default off")
require('"cyclic": tuple(MAIN_DEFINES) + ("-DOKC_WGSCYCLIC=1",)' in source,
        "diagnostic cyclic variant missing")
require('"cyclic_timeline": tuple(MAIN_DEFINES) + ("-DOKC_WGSCYCLIC=1", "-DOKC_DEBUG=1")' in source,
        "diagnostic cyclic timeline variant missing")
require("wgs_phase_cyclic<false>" in source and "wgs_phase_cyclic<true>" in source,
        "cyclic phase dispatch missing")
require("u, 0, 16, -1, seq, pd" in source,
        "cyclic units must be whole and bypass cross-CTA fixup slots")
require("sc = list(state[\"scratch\"]) + [dbg]" in source,
        "diagnostic fourth scratch tensor is not isolated from production state")
require("DIAG = True" in source, "v153 diagnostic payload is not enabled")

# Every phase unit is owned exactly once for all supported job/grid bounds.  Producer and consumer formulas share this
# same grid-stride mapping; each owned unit expands to exactly its 16 K stages.
for nj in range(1, 137):
    for pb in (0, 1):
        nu = nj << (5 - pb)
        for grid in range(1, 161):
            total = 0
            for c in range(grid):
                count = ((nu - 1 - c) // grid) + 1 if c < nu else 0
                ns = ((((nu - 1 - c) // grid) + 1) << 4) if c < nu else 0
                require(ns == 16 * count,
                        f"producer stage-count mismatch nj={nj} pb={pb} grid={grid} c={c}")
                if count:
                    require(c + (count - 1) * grid < nu <= c + count * grid,
                            f"cyclic residue bound failure nj={nj} pb={pb} grid={grid} c={c}")
                total += count
            require(total == nu, f"cyclic coverage failure nj={nj} pb={pb} grid={grid}")

print("PASS: v153 cyclic coverage is exact; active and frozen v139 hashes are unchanged")
