"""Restore protected S052 after S075 failed the whole-layer promotion gate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "submission.py"
ROLLBACK = ROOT / "submission_s052_large_b4_partial_norm.py"

S075_SHA256 = "398dfa6195156898c51c29ac87464a77d8a7ff539f01b9f9b3b581be19b77b55"
S052_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(ACTIVE) != S075_SHA256:
    raise RuntimeError("active submission is not the promoted S075 candidate")
if sha(ROLLBACK) != S052_SHA256:
    raise RuntimeError("protected S052 rollback hash mismatch")

temp = ROOT / "submission.py.s052.tmp"
temp.write_bytes(ROLLBACK.read_bytes())
if sha(temp) != S052_SHA256:
    raise RuntimeError("temporary rollback copy failed hash verification")
os.replace(temp, ACTIVE)
print(ACTIVE)
print(sha(ACTIVE))
