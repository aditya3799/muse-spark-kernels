"""Fail-closed audit for the scored v178 live promotion."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
WINNER = ROOT / "submission_v178_hybrid_m32_retry.py"
BACKUP = ROOT / "submission_pre_v178_live.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
EXPECTED_WINNER = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"
EXPECTED_OLD = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"
EXPECTED_FROZEN = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    assert sha(LIVE) == EXPECTED_WINNER
    assert sha(WINNER) == EXPECTED_WINNER
    assert LIVE.read_bytes() == WINNER.read_bytes()
    assert sha(BACKUP) == EXPECTED_OLD
    assert sha(FROZEN) == EXPECTED_FROZEN
    ast.parse(LIVE.read_text())
    print("live submission equals exact scored v178 winner: PASS")
    print("pre-v178 live backup hash: PASS")
    print("frozen v139 hash unchanged: PASS")
    print(f"live SHA256={EXPECTED_WINNER}")


if __name__ == "__main__":
    main()
