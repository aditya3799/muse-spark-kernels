"""Materialize the complete pinned chenwainuo Muse BF16 projection submission.

This is a byte-for-byte research candidate from the public Apache-2.0 source.
The SHA is pinned so a future upstream change cannot silently alter the file.
"""

from __future__ import annotations

import ast
import hashlib
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission_chenwainuo_full_v167.py"
CACHE = Path(tempfile.gettempdir()) / "chenwainuo_3d418e1_submission.py"
URL = (
    "https://raw.githubusercontent.com/chenwainuo/okc/3d418e1/"
    "muse_bf16_proj/submission.py"
)
EXPECTED = "BB50079BE2ED74D65BB7109F1432FBC7A49BBE70913D247C254717CD0F27F712"


if CACHE.exists():
    data = CACHE.read_bytes()
else:
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read()
    CACHE.write_bytes(data)

actual = hashlib.sha256(data).hexdigest().upper()
if actual != EXPECTED:
    raise RuntimeError(f"upstream SHA256 mismatch: {actual}")
if not data.startswith(b"# okc: problem=muse_bf16_proj"):
    raise RuntimeError("upstream problem marker missing")
ast.parse(data.decode())
OUT.write_bytes(data)
print(OUT.name, actual, len(data))
