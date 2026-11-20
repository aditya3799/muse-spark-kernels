"""Build S040 from team vjeux's public Apache-2.0 whole-layer entry."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(tempfile.gettempdir()) / "vjeux_muse_decode_layer_f35a41ee.py"
DESTINATION = ROOT / "submission_s040_vjeux_f35a41ee.py"
SOURCE_SHA256 = "cf4a75ee6e4a5cd9e33b82b5347c47abcdc7b81d80dc6a8b5262dea7388f849c"
HEADER = "# okc: problem=muse_decode_layer\n"
ATTRIBUTION = (
    "# Ported unchanged from team vjeux's public Open Kernel Challenge entry,\n"
    "# commit f35a41ee1c13e316dabbf772830d47b23b453393, Apache-2.0.\n"
)


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected pinned vjeux source hash: {actual_source}")
source = source_bytes.decode("utf-8")
if not source.startswith(HEADER):
    raise RuntimeError("pinned source lost the decoder problem marker")
candidate = HEADER + ATTRIBUTION + source[len(HEADER):]

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S040 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
