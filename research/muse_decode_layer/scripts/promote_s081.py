"""Promote the exact officially checked S081 bytes under strict hash guards."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "submission.py"
BACKUP = ROOT / "submission_s052_large_b4_partial_norm.py"
CANDIDATE = ROOT / "submission_s081_keell0renz_890809c.py"

S052_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"
S081_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(ACTIVE) != S052_SHA256:
    raise RuntimeError("active submission is not protected S052; refusing overwrite")
if sha(BACKUP) != S052_SHA256:
    raise RuntimeError("byte-identical S052 rollback file is missing")
if sha(CANDIDATE) != S081_SHA256:
    raise RuntimeError("officially checked S081 candidate hash mismatch")

temp = ROOT / "submission.py.s081.tmp"
temp.write_bytes(CANDIDATE.read_bytes())
if sha(temp) != S081_SHA256:
    raise RuntimeError("temporary promotion copy failed hash verification")
os.replace(temp, ACTIVE)
print(f"promoted S081 to {ACTIVE}")
print("sha256", sha(ACTIVE))
