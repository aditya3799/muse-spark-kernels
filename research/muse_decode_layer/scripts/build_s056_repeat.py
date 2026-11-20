"""Create a semantically identical S056 file for an independent OKC check."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s056_joined_wgs_completion.py"
DESTINATION = ROOT / "submission_s056r_joined_wgs_completion.py"
SOURCE_SHA256 = "e19c8b10fd6585b9c2d69e6c40efc3584036ea1ed57f3a0b982806c260b1751d"


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S056 hash: {actual_source}")

candidate = source_bytes.decode("utf-8") + "\n# S056 independent measurement replica; kernel and dispatch are unchanged.\n"
ast.parse(candidate)
encoded = candidate.encode("utf-8")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
