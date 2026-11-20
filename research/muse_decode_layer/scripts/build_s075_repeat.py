"""Create a semantics-identical, byte-distinct S075 repeat-check artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s075_cgemv_rs128.py"
OUT = ROOT / "submission_s075r_cgemv_rs128.py"
SOURCE_SHA256 = "398dfa6195156898c51c29ac87464a77d8a7ff539f01b9f9b3b581be19b77b55"

data = SOURCE.read_bytes()
if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("validated S075 source hash mismatch")
data += b"\n# S075 repeat-check artifact; semantics identical.\n"
compile(data.decode(), str(OUT), "exec")
OUT.write_bytes(data)
print(OUT)
print(hashlib.sha256(data).hexdigest())
