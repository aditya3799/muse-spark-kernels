"""Pin the archived winning legal-BF16 M=8192 unpooler cuBLASLt plan."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
src = root / "submission_n72_exactroute_clean_candidate.py"
out = root / "submission_n72_exactroute_pin8k_candidate.py"
text = src.read_text()

old = '''LT_PIN = {(753, "router"): (33, 4, 2, 3), (16384, "o"): (201, 1, 0, 2), (16384, "unpooler"): (201, 1, 0, 2),
          (16384, "pooler"): (29, 1, 0, 1)}'''
new = '''LT_PIN = {(753, "router"): (33, 4, 2, 3), (8192, "unpooler"): (201, 1, 0, 2),
          (16384, "o"): (201, 1, 0, 2), (16384, "unpooler"): (201, 1, 0, 2),
          (16384, "pooler"): (29, 1, 0, 1)}'''
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
