"""Fail-closed audit for the protected-v178 research factorial diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from make_diag_v184_research_factorial import APPEND


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
WINNER = ROOT / "submission_v178_hybrid_m32_retry.py"
DIAG = ROOT / "submission_diag_v184_research_factorial.py"
WINNER_SHA = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    assert sha(LIVE) == WINNER_SHA, ("live changed", sha(LIVE))
    assert sha(WINNER) == WINNER_SHA, ("winner changed", sha(WINNER))
    expected = WINNER.read_bytes().rstrip(b"\r\n") + APPEND.encode("utf-8")
    assert DIAG.read_bytes() == expected
    text = DIAG.read_text()
    ast.parse(text)
    assert '_RF_SEEDS = 5' in text and '_RF_REPS = 16' in text
    assert '"control", "grid_plain", "mfin_plain", "grid_sc6", "mfin_sc6"' in text
    assert 'missing = ("grid_plain", "mfin_plain", "grid_sc6")' in text
    assert 'state = _RF_PARENT_PREPARE(weights)' in text
    assert 'for M in (128, 256):' in text
    assert 'verify=' in text and 'projected-score=' in text
    assert 'vr <= 0.985 and win_count >= 4' in text
    assert '@v167_main' in text and '-DOKC_MFIN=0' in text and '-DOKC_SC6W=1' in text
    print("protected v178 + isolated five-cell M128/M256 factorial: PASS")
    print(f"diagnostic SHA256={sha(DIAG)}")


if __name__ == "__main__":
    main()
