"""Promote the scored v178 winner to submission.py with an exact backup."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
WINNER = ROOT / "submission_v178_hybrid_m32_retry.py"
BACKUP = ROOT / "submission_pre_v178_live.py"
EXPECTED_OLD = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"
EXPECTED_WINNER = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> None:
    old = LIVE.read_bytes()
    winner = WINNER.read_bytes()
    assert sha(old) == EXPECTED_OLD, (sha(old), EXPECTED_OLD)
    assert sha(winner) == EXPECTED_WINNER, (sha(winner), EXPECTED_WINNER)
    if BACKUP.exists():
        assert sha(BACKUP.read_bytes()) == EXPECTED_OLD
    else:
        BACKUP.write_bytes(old)
    LIVE.write_bytes(winner)
    assert sha(LIVE.read_bytes()) == EXPECTED_WINNER
    print(f"backed up old live source as {BACKUP.name}")
    print(f"promoted {WINNER.name} to {LIVE.name}")
    print(f"live SHA256={EXPECTED_WINNER}")


if __name__ == "__main__":
    main()
