"""Promote the scored stage5+W0 winner after immutable hash checks."""

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_sweep_stage5_w0_current.py"
DST = ROOT / "submission.py"
EXPECTED_SRC = "3b4e9a448efb719021334cb664505d9a0412a70202776acfdc6971c2088aeeb8"
EXPECTED_DST = "a644b2aff0a5299bccf01f39586df25ddc13d37af5a63b0871dd1c92b088772e"

src_bytes = SRC.read_bytes()
dst_bytes = DST.read_bytes()
if sha256(src_bytes).hexdigest() != EXPECTED_SRC:
    raise SystemExit("finalist hash changed; refusing promotion")
if sha256(dst_bytes).hexdigest() != EXPECTED_DST:
    raise SystemExit("incumbent hash changed; refusing promotion")

DST.write_bytes(src_bytes)
print(f"promoted {SRC.name} -> {DST.name}: {sha256(src_bytes).hexdigest()}")
