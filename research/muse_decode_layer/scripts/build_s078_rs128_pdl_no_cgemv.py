"""Build S078: retain S077's T=128 RS128/PDL path without S073 T=1 CGEMV.

S077 was composed through S075, which appended the independent S073 exact-C12
T=1 experiment before the RS128 pooler/unpooler and PDL work.  The scored S077
result showed a useful T=128 gain but a slightly worse whole-suite aggregate.
This controlled candidate removes the complete S073 appendix while preserving
every other byte of S077, including the validated T=128 projection plans and
the programmatic-dependent-launch implementation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_s077_rs128_pdl.py"
OUT = ROOT / "submission_s078_rs128_pdl_no_cgemv.py"

BASE_SHA256 = "437eada9f1ef0964b69fb804cadfe392e8adfbac72a98be65d75723a00d72178"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("validated S077 hash mismatch; refusing to derive S078")

base = BASE.read_text(encoding="utf-8")
s073_marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
s076_marker = "# S076: complete the protected bf16_proj v150 RS128 transfer at T=128."

s073_pos = base.index(s073_marker)
s076_pos = base.index(s076_marker)
begin = base.rfind("# ---------------------------------------------------------------------------", 0, s073_pos)
end = base.rfind("# ---------------------------------------------------------------------------", 0, s076_pos)
if begin < 0 or end <= begin:
    raise RuntimeError("could not isolate the S073 appendix")

candidate = base[:begin].rstrip() + "\n\n" + base[end:].lstrip()
if "_S073_CG_" in candidate or "s073_cg" in candidate:
    raise RuntimeError("S073 runtime symbols remain after removal")
if "PDL_MAX_T = 128" not in candidate or s076_marker not in candidate:
    raise RuntimeError("S077 T=128 path was not preserved")

candidate += """

# ---------------------------------------------------------------------------
# S078: S077's validated T=128 RS128/PDL path with the independent S073 T=1
# exact-C12 CGEMV appendix removed for a controlled whole-suite comparison.
# ---------------------------------------------------------------------------
"""

compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
