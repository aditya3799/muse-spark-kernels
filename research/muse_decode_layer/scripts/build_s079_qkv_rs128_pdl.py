"""Build S079: isolate QKV RS128/PDL and restore S052 pool/unpool routes.

S077 contains three independently appended layers: S073 T=1 CGEMV, S076
RS128 pooler/unpooler, and the S077 PDL conversion applied to the underlying
S074 QKV candidate.  This builder removes the first two appendices as one
contiguous block.  The result keeps the scored QKV RS128/PDL implementation
and T=128 PDL chain, while pooler/router and unpooler fall back to S052's
original fused/tuned implementations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_s077_rs128_pdl.py"
OUT = ROOT / "submission_s079_qkv_rs128_pdl.py"

BASE_SHA256 = "437eada9f1ef0964b69fb804cadfe392e8adfbac72a98be65d75723a00d72178"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("validated S077 hash mismatch; refusing to derive S079")

base = BASE.read_text(encoding="utf-8")
s073_marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
s077_marker = "# S077: experimental v150 RS128 programmatic dependent launch at T=128."

s073_pos = base.index(s073_marker)
s077_pos = base.index(s077_marker)
begin = base.rfind("# ---------------------------------------------------------------------------", 0, s073_pos)
end = base.rfind("# ---------------------------------------------------------------------------", 0, s077_pos)
if begin < 0 or end <= begin:
    raise RuntimeError("could not isolate the S073/S076 appendices")

candidate = base[:begin].rstrip() + "\n\n" + base[end:].lstrip()
for forbidden in ("_S073_CG_", "s073_cg", "_s076_", "s076_rs128"):
    if forbidden in candidate:
        raise RuntimeError(f"removed runtime symbol remains: {forbidden}")
required = (
    "PDL_MAX_T = 128",
    "def _s074_rs128_run",
    'return _v150_call(vstate, vstate["plan"], x, pdl=True)',
    s077_marker,
)
for fragment in required:
    if fragment not in candidate:
        raise RuntimeError(f"required QKV/PDL fragment missing: {fragment}")

candidate += """

# ---------------------------------------------------------------------------
# S079: QKV-only RS128/PDL at T=128.  Pooler/router and unpooler use the
# original S052 paths so their fusion and in-chain PDL behavior can be tested.
# ---------------------------------------------------------------------------
"""

compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
