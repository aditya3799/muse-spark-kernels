"""Build an exact-route hybrid that keeps lossless K64 only at T753."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
src = root / "submission_n72_exactroute_clean_candidate.py"
out = root / "submission_n72_exactroute_smallkperm_candidate.py"
text = src.read_text()

old = "KPERM_LARGE_MIN_T = 8192  # experiment: FlashInfer-style K64 permutation on the dominant 16K workload"
new = "KPERM_LARGE_MIN_T = 0  # retain the original lossless layout at large T; K64 stays enabled only for T753"
assert text.count(old) == 1
text = text.replace(old, new, 1)

for forbidden in (
    "ROUTE_SHAPING",
    "ROUTE_SHAPE_MIN_T",
    "route_shape_",
    "candsc",
    "checker-optional",
    "base64.b64decode",
    "zlib.decompress",
):
    assert forbidden not in text, forbidden
assert "UNPOOL_FP8 = False" in text and "O_FP8 = False" in text
ast.parse(text)
compile(text, str(out), "exec")
out.write_bytes(text.encode())
print(out.name, len(text.encode()))
