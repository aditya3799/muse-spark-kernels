"""Promote the scored q8-W0 winner after immutable hash checks."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_sweep_q8_w0.py"
DST = ROOT / "submission.py"
EXPECTED_SRC = "3e6d676275706a76c624b27a970583e6157a8cc536c4843d72fdc40b986172a2"
EXPECTED_DST = "a644b2aff0a5299bccf01f39586df25ddc13d37af5a63b0871dd1c92b088772e"

src_bytes = SRC.read_bytes()
dst_bytes = DST.read_bytes()
if sha256(src_bytes).hexdigest() != EXPECTED_SRC:
    raise SystemExit("q8 W0 candidate hash changed; refusing promotion")
if sha256(dst_bytes).hexdigest() != EXPECTED_DST:
    raise SystemExit("incumbent hash changed; refusing promotion")

DST.write_bytes(src_bytes)
print(f"promoted {SRC.name} -> {DST.name}: {sha256(src_bytes).hexdigest()}")
