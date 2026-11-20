"""Prove that v185 differs from protected v178 only in two dispatch predicates."""

import ast
import difflib
import hashlib
from pathlib import Path


ROOT = Path(__file__).parent
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
CANDIDATE = ROOT / "submission_v185_hybrid_m32_m128_m256.py"
PROTECTED_SHA = "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = BASE.read_text()
    candidate = CANDIDATE.read_text()
    assert sha(BASE) == PROTECTED_SHA
    ast.parse(candidate)
    diff = list(difflib.unified_diff(base.splitlines(), candidate.splitlines(), lineterm=""))
    changes = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert changes == [
        '-    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]',
        '+    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if (_M <= 32 or _M in (128, 256)) else state["larger"]',
        '-    elif M <= 32:',
        '+    elif M <= 32 or M in (128, 256):',
    ], changes
    assert sha(ROOT / "submission.py") == PROTECTED_SHA
    print(f"protected={PROTECTED_SHA}")
    print(f"candidate={sha(CANDIDATE)}")
    print("v185 narrow M128/M256 dispatch audit: PASS")


if __name__ == "__main__":
    main()
