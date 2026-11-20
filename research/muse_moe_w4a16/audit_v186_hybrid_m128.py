"""Prove that v186 changes only the M128 state/module dispatch."""

import ast
import difflib
import hashlib
from pathlib import Path


ROOT = Path(__file__).parent
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
CANDIDATE = ROOT / "submission_v186_hybrid_m32_m128.py"
PROTECTED_SHA = "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = BASE.read_text()
    candidate = CANDIDATE.read_text()
    assert sha(BASE) == PROTECTED_SHA
    assert sha(ROOT / "submission.py") == PROTECTED_SHA
    ast.parse(candidate)
    diff = list(difflib.unified_diff(base.splitlines(), candidate.splitlines(), lineterm=""))
    changes = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert changes == [
        '-    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]',
        '+    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if (_M <= 32 or _M == 128) else state["larger"]',
        '-    elif M <= 32:',
        '+    elif M <= 32 or M == 128:',
    ], changes
    print(f"protected={PROTECTED_SHA}")
    print(f"candidate={sha(CANDIDATE)}")
    print("v186 M128-only dispatch audit: PASS")


if __name__ == "__main__":
    main()
