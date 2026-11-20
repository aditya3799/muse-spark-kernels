"""Build the maximal gated exact-route candidate on the small-only K64 base."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
src = root / "submission_n72_exactroute_lt2fixed_candidate.py"
out = root / "submission_n72_exactroute_smallkperm_pin8k_lt2fixed_candidate.py"
text = src.read_text()

old = "KPERM_LARGE_MIN_T = 8192  # experiment: FlashInfer-style K64 permutation on the dominant 16K workload"
new = "KPERM_LARGE_MIN_T = 0  # use lossless K64 only at T753; retain the standard large-T layout"
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
assert '(8192, "unpooler"): (201, 1, 0, 2)' in text
assert "LT_ALGO_TOL = 2e-2" in text
ast.parse(text)
compile(text, str(out), "exec")
out.write_bytes(text.encode())
print(out.name, len(text.encode()))
