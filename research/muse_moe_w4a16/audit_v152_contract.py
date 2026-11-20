"""Static contract audit for the diagnostics-only v152 WGS wait profiler."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "submission_v139_wgs_kperm3.py"
CANDIDATE = ROOT / "submission_v152_wgs_waitprof_diag.py"
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
require("#define OKC_WGSPROF 0" in source, "wait profiler must default off")
require('"prodprof": tuple(MAIN_DEFINES) + ("-DOKC_DEBUG=1", "-DOKC_WGSPROF=3")' in source,
        "diagnostic producer-profiler variant missing")
require("WPROF_DONE_AS" in source and "WPROF_X_READY" in source and "[waitprof:" in source,
        "required producer-readiness instrumentation missing")
require("sc = list(state[\"scratch\"]) + [dbg]" in source,
        "diagnostic fourth scratch tensor is not isolated from production state")
require("DIAG = True" in source, "v152 diagnostic payload is not enabled")

print("PASS: v152 instrumentation is diagnostics-only; active and frozen v139 hashes are unchanged")
