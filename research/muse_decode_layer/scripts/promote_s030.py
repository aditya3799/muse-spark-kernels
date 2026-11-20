"""Promote the exact checked S030 candidate to submission.py with hash guards."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "submission.py"
CANDIDATE = ROOT / "submission_s030_v81_m32.py"
CURRENT_SHA256 = "f6c081ce2ce356e194b9a163a4ccab92b4b530da3a564ae56e5ed147b405786c"
CANDIDATE_SHA256 = "4b40bd21366077da9a3d969b5ca876fc67db308767fad60e7bc535cab111b2b6"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


current = CURRENT.read_bytes()
candidate = CANDIDATE.read_bytes()
if sha256(current) != CURRENT_SHA256:
    raise RuntimeError(f"submission.py is not the expected S022: {sha256(current)}")
if sha256(candidate) != CANDIDATE_SHA256:
    raise RuntimeError(f"S030 candidate hash changed: {sha256(candidate)}")
if not candidate.startswith(b"# okc: problem=muse_decode_layer\n"):
    raise RuntimeError("S030 lost the required problem marker")
if len(candidate) >= 1_000_000:
    raise RuntimeError("S030 exceeds the source-size limit")
ast.parse(candidate.decode("utf-8"))
CURRENT.write_bytes(candidate)
print(f"promoted {CANDIDATE.name} to {CURRENT}")
print("sha256", sha256(CURRENT.read_bytes()))
