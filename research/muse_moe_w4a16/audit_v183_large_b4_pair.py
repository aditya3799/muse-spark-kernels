"""Fail-closed audit for the v183 production candidate plus diagnostic appendix."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from audit_v183_hybrid_large_b4 import main as audit_candidate
from make_diag_v183_large_b4_pair import APPEND


ROOT = Path(__file__).resolve().parent
PROD = ROOT / "submission_v183_hybrid_large_b4.py"
DIAG = ROOT / "submission_diag_v183_large_b4_pair.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    audit_candidate()
    assert PROD.read_bytes().rstrip(b"\r\n") + APPEND.encode("utf-8") == DIAG.read_bytes()
    text = DIAG.read_text()
    ast.parse(text)
    assert "_LB4_SEEDS = 5" in text and "_LB4_REPS = 20" in text
    assert 'control = state["larger"]' in text
    assert 'candidate = state["large_b4"]' in text
    assert 'for M in (128, 256):' in text
    assert 'projected-score-ratio=' in text
    assert 'assert max(rels) < 1e-2' in text
    print("five-seed paired M128/M256 diagnostic exactly appended: PASS")
    print(f"diagnostic SHA256={sha(DIAG)}")


if __name__ == "__main__":
    main()
