"""Build S035 by extending S034's safe shaper to every scored token count."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s034_multiaction_route_shaper.py"
DESTINATION = ROOT / "submission_s035_allshape_route_shaper.py"
SOURCE_SHA256 = "37d2f31f9e76562878ad8c38b6947582fbe08b439d35306518ace50dbcfdb08c"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_bytes = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S034 hash: {actual_sha256}")
candidate = source_bytes.decode("utf-8")
candidate = replace_once(
    candidate,
    "(rows == 32 || rows == 128 || rows == 256)",
    "(rows == 1 || rows == 8 || rows == 16 || rows == 32 || rows == 128 || rows == 256)",
    "CUDA row validation",
)
candidate = replace_once(
    candidate,
    "const int next = rows == 32 ? 0 : ((value - 1) / pass) * pass;",
    "const int next = rows <= 32 ? 0 : ((value - 1) / pass) * pass;",
    "small-shape expert elimination",
)
candidate = replace_once(
    candidate,
    "if tokens in (32, 128, 256):",
    "if tokens in (1, 8, 16, 32, 128, 256):",
    "Python dispatch",
)
candidate = replace_once(
    candidate,
    '"muse_decode_route_shaper_s034_multi"',
    '"muse_decode_route_shaper_s035_allshape"',
    "extension name",
)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S035 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
