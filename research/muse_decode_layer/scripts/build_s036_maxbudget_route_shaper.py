"""Build S036: S035 using the exact published locality budgets."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s035_allshape_route_shaper.py"
DESTINATION = ROOT / "submission_s036_maxbudget_route_shaper.py"
SOURCE_SHA256 = "9504764fe4e25f0e7e49c115b4e3ce9f4afe6dfc1542f493e36adbabe77126d7"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_bytes = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S035 hash: {actual_sha256}")
candidate = source_bytes.decode("utf-8")
candidate = replace_once(
    candidate,
    "int budget = total * 7 / 100 + 3;",
    "int budget = total * 8 / 100 + 4;",
    "local-route budget",
)
candidate = replace_once(
    candidate,
    "const int max_actions = max(1, distinct * 12 / 100 + 1);",
    "const int max_actions = max(1, distinct * 15 / 100 + 2);",
    "local-distinct budget",
)
candidate = replace_once(
    candidate,
    '"muse_decode_route_shaper_s035_allshape"',
    '"muse_decode_route_shaper_s036_maxbudget"',
    "extension name",
)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S036 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
