"""Promote scored winner 07e5d76c9c114a4b while preserving the old incumbent."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "submission.py"
CANDIDATE = ROOT / "submission_recovered_cf1a.py"
BACKUP = ROOT / "submission_incumbent_2607619x.py"

EXPECTED_ACTIVE = "1edef4d178cdbee3b256ba3d662738a0f5ba7788c1782abcf012341f8a3f20c9"
EXPECTED_CANDIDATE = "a644b2aff0a5299bccf01f39586df25ddc13d37af5a63b0871dd1c92b088772e"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if digest(ACTIVE) != EXPECTED_ACTIVE:
    raise SystemExit(f"active source changed unexpectedly: {digest(ACTIVE)}")
if digest(CANDIDATE) != EXPECTED_CANDIDATE:
    raise SystemExit(f"candidate source changed unexpectedly: {digest(CANDIDATE)}")

old = ACTIVE.read_text(encoding="utf-8")
new = CANDIDATE.read_text(encoding="utf-8")
BACKUP.write_text(old, encoding="utf-8", newline="\n")
ACTIVE.write_text(new, encoding="utf-8", newline="\n")
print(f"backup {BACKUP} {digest(BACKUP)}")
print(f"active {ACTIVE} {digest(ACTIVE)}")
