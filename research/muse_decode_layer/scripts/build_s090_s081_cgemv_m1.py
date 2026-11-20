"""Append the already validated S073 T=1 GEMV graft to protected S081.

The graft is self-contained and wraps the final prepare/kernel entry points, so
the S081 attention, routing, MoE, and all non-T=1 projection paths stay exact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "submission.py"
SOURCE = ROOT / "submission_s073_cgemv_m1.py"
OUT = ROOT / "submission_s090_s081_cgemv_m1.py"

ACTIVE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"
SOURCE_SHA256 = "027d4abd1f6da65c6511f86a3e735d5cff5e7a379a02f8f944e245125b6b1161"
MARKER = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(ACTIVE) != ACTIVE_SHA256:
    raise RuntimeError("protected S081 hash mismatch")
if sha(SOURCE) != SOURCE_SHA256:
    raise RuntimeError("validated S073 source hash mismatch")

base = ACTIVE.read_text(encoding="utf-8").rstrip()
source = SOURCE.read_text(encoding="utf-8")
marker_pos = source.index(MARKER)
appendix_pos = source.rfind("# ---------------------------------------------------------------------------", 0, marker_pos)
if appendix_pos < 0:
    raise RuntimeError("S073 appendix boundary missing")

candidate = base + "\n\n" + source[appendix_pos:].rstrip() + "\n"
compile(candidate, str(OUT), "exec")
OUT.write_text(candidate, encoding="utf-8", newline="\n")
print(OUT)
print(sha(OUT))
