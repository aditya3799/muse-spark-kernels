"""Build S031 by enabling the validated v81 verify-attention kernel for B1.

S029 repaired the wrapper's input rank by passing a 3D ``(B, L, D)`` view.
The embedded v81 source and configuration already contain B1 specializations;
only the conservative support predicate still excluded them.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.py"
DESTINATION = ROOT / "submission_s031_v81_b1.py"
SOURCE_SHA256 = "4b40bd21366077da9a3d969b5ca876fc67db308767fad60e7bc535cab111b2b6"


source_bytes = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(
        f"unexpected S030 source hash: {actual_sha256} != {SOURCE_SHA256}"
    )

candidate = source_bytes.decode("utf-8")
old = '        and cfg["batch"] in (16, 32)\n'
new = '        and cfg["batch"] in (1, 16, 32)\n'
if candidate.count(old) != 1:
    raise RuntimeError(f"expected one support-predicate match, found {candidate.count(old)}")
candidate = candidate.replace(old, new, 1)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S031 exceeds the challenge source limit")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
