"""Promote the validated S075 candidate to submission.py with hash guards."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "submission.py"
BACKUP = ROOT / "submission_s052_large_b4_partial_norm.py"
CANDIDATE = ROOT / "submission_s075_cgemv_rs128.py"

S052_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"
S075_SHA256 = "398dfa6195156898c51c29ac87464a77d8a7ff539f01b9f9b3b581be19b77b55"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(ACTIVE) != S052_SHA256:
    raise RuntimeError("active submission is not protected S052; refusing overwrite")
if sha(BACKUP) != S052_SHA256:
    raise RuntimeError("byte-identical S052 rollback file is missing")
if sha(CANDIDATE) != S075_SHA256:
    raise RuntimeError("validated S075 candidate hash mismatch")

data = CANDIDATE.read_bytes()
temp = ROOT / "submission.py.s075.tmp"
temp.write_bytes(data)
if sha(temp) != S075_SHA256:
    raise RuntimeError("temporary promotion copy failed hash verification")
os.replace(temp, ACTIVE)
print(ACTIVE)
print(sha(ACTIVE))
