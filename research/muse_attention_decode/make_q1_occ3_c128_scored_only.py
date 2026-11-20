"""Add register-pool headroom to the isolated 3-CTA q_len=1 candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_q1_occ3_scored_only.py"
DEST = ROOT / "submission_q1_occ3_c128_scored_only.py"
EXPECTED_SOURCE = "acfe6c4e8415ba845f839ef095c6202d8f95eb6123691e761b2775eabac8f8ab"


raw = SOURCE.read_bytes()
got = hashlib.sha256(raw).hexdigest()
if got != EXPECTED_SOURCE:
    raise RuntimeError(f"isolated q1-occ3 source hash mismatch: {got}")

text = raw.decode().replace("\r\n", "\n")
old = "    src = CUDA_SRC\n"
new = ("    src = CUDA_SRC\n"
       "    # PREGS=72 makes the derived q1-flat consumer request 128 registers.\n"
       "    # Three CTAs then request 62,976 of 65,536 registers, leaving 2,560\n"
       "    # registers of headroom instead of sitting on the allocation cliff.\n"
       "    preg_old = \"#define OKC_PREGS 0\"\n"
       "    preg_new = \"#define OKC_PREGS 72\"\n"
       "    if src.count(preg_old) != 1:\n"
       "        raise RuntimeError(\"q1-occ3-c128: PREGS source anchor changed\")\n"
       "    src = src.replace(preg_old, preg_new, 1)\n")
if text.count(old) != 1:
    raise RuntimeError("could not locate isolated module source construction")
text = text.replace(old, new, 1)

old_name = "okc_attn_v298_q1occ3_scored_90a"
new_name = "okc_attn_v298_q1occ3c128_scored_90a"
if text.count(old_name) != 1:
    raise RuntimeError("could not locate isolated module build name")
text = text.replace(old_name, new_name, 1)

text = text.replace(
    "# q1-occ3 scored-only diagnostic:",
    "# q1-occ3-c128 scored-only diagnostic:",
    1,
)
DEST.write_text(text)
print(DEST)
print(hashlib.sha256(DEST.read_bytes()).hexdigest())
