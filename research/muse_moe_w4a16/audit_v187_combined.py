"""Audit v187 as the exact composition of two previously paired modules."""

import ast
import difflib
import hashlib
from pathlib import Path


ROOT = Path(__file__).parent
BASE = ROOT / "submission_v182_hybrid_m32_decoder_b4.py"
CANDIDATE = ROOT / "submission_v187_combined_b4_large.py"
LIVE = ROOT / "submission.py"
BASE_SHA = "93274ec0657cce5ff1c50e3e241790324c6bbac89dcd06777ac43505371da365"
LIVE_SHA = "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = BASE.read_text()
    candidate = CANDIDATE.read_text()
    assert sha(BASE) == BASE_SHA
    assert sha(LIVE) == LIVE_SHA
    ast.parse(candidate)
    diff = list(difflib.unified_diff(base.splitlines(), candidate.splitlines(), lineterm=""))
    changes = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert changes == [
        '+    _ext(tuple(V167_MAIN_DEFINES) + state["sfx"] + ("@v167_main",), use_mul=False)',
        '-    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]',
        '+    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if (_M <= 32 or _M in (128, 256)) else state["larger"]',
        '+    elif M in (128, 256):',
        '+        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)',
    ], changes
    assert candidate.count('(\"-DOKC_B4FIN=1\", \"@v182_decoder_b4\")') == 2
    assert candidate.count('(\"@v167_main\",), use_mul=False') >= 2
    print(f"live={LIVE_SHA}")
    print(f"candidate={sha(CANDIDATE)}")
    print("v187 exact B4 + MFIN/SC6 composition audit: PASS")


if __name__ == "__main__":
    main()
