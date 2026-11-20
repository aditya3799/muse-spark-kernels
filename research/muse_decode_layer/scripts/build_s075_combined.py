"""Build S075 by composing the independently validated S073 and S074 grafts.

The mechanisms are disjoint at runtime: exact-C12 GEMV is restricted to T=1,
while RS128 is restricted to T=128 QKV.  Their generated inputs are hash-pinned
so the composition cannot silently pick up a different candidate revision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CG = ROOT / "submission_s073_cgemv_m1.py"
RS = ROOT / "submission_s074_rs128_qkv.py"
OUT = ROOT / "submission_s075_cgemv_rs128.py"

CG_SHA256 = "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161"
RS_SHA256 = "d0c02e4924a73f2e3f93c54226e9b62303213276ac9895ed6c7a7ef8db8e31b0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(CG) != CG_SHA256:
    raise RuntimeError("validated S073 hash mismatch; refusing to compose")
if _sha(RS) != RS_SHA256:
    raise RuntimeError("validated S074 hash mismatch; refusing to compose")

cg = CG.read_text(encoding="utf-8")
rs = RS.read_text(encoding="utf-8")
marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
pos = cg.index(marker)
begin = cg.rfind("# ---------------------------------------------------------------------------", 0, pos)
if begin < 0:
    raise RuntimeError("S073 appendix marker missing")
appendix = cg[begin:].rstrip()

candidate = rs.rstrip() + "\n\n" + appendix + "\n"
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
