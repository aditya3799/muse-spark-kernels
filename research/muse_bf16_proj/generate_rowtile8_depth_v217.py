"""Generate v217: 8-row QKV retile with incumbent five-stage rings."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V215 = ROOT / "submission_plain_rowtile8_qkv_v215.py"
OUTPUT = ROOT / "submission_plain_rowtile8_depth_v217.py"

text = BASE.read_text(encoding="utf-8")
source = V215.read_text(encoding="utf-8")
marker = "# v215: retile QKV from 24 to 8 rows per CTA"
if source.count(marker) != 1:
    raise RuntimeError("v215 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v215", "v217").replace("V215", "V217")

replacements = {
    "v217_q8_r8a_r144": "v217_q8_r8a_r72",
    "v217_q8_r8b_r144": "v217_q8_r8b_r72",
    "v217_q16_r8a_r200": "v217_q16_r8a_r112",
    "v217_q16_r8b_r200": "v217_q16_r8b_r112",
}
for old, new in replacements.items():
    if tail.count(old) != 1:
        raise RuntimeError(f"v217 config marker mismatch: {old}")
    tail = tail.replace(old, new, 1)
tail = tail.replace(
    "# v217: retile QKV from 24 to 8 rows per CTA (96 to 288 CTAs).  Test\n"
    "# both four-K-warp and two-K-warp decompositions; arithmetic and exact C12\n"
    "# storage stay unchanged and paired cold-L2 replay gates every route.\n",
    "# v217: repeat the 8-row QKV retile with byte caps that retain the\n"
    "# incumbent five-stage ring.  Both K-warp decompositions remain exact and\n"
    "# paired cold-L2 replay gates every route independently.\n",
    1)
tail = tail.replace("v217 rowtile8", "v217 rowtile8-depth")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
