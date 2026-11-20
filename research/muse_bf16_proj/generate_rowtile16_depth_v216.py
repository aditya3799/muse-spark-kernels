"""Generate v216: 16-row retile with incumbent-equivalent ring depths."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V214 = ROOT / "submission_plain_rowtile16_small_v214.py"
OUTPUT = ROOT / "submission_plain_rowtile16_depth_v216.py"

text = BASE.read_text(encoding="utf-8")
source = V214.read_text(encoding="utf-8")
marker = "# v214: retile QKV from 24 to 16 rows per CTA"
if source.count(marker) != 1:
    raise RuntimeError("v214 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v214", "v216").replace("V214", "V216")

# With fewer rows each ring stage is smaller.  Lower the byte caps so this pass
# preserves the incumbent pipeline depth instead of v214's accidental 6/7-stage
# expansion; this isolates CTA flight count from ring depth.
replacements = {
    "v216_q8_r16a_r144": "v216_q8_r16a_r100",
    "v216_q8_r16b_r144": "v216_q8_r16b_r100",
    "v216_q16_r16a_r200": "v216_q16_r16a_r140",
    "v216_q16_r16b_r200": "v216_q16_r16b_r140",
    "v216_p16_r16_r192": "v216_p16_r16_r128",
    "v216_p32_r16_r144": "v216_p32_r16_r112",
}
for old, new in replacements.items():
    if tail.count(old) != 1:
        raise RuntimeError(f"v216 config marker mismatch: {old}")
    tail = tail.replace(old, new, 1)
tail = tail.replace(
    "# v216: retile QKV from 24 to 16 rows per CTA (96 to 144 CTAs), and\n"
    "# pooler from 32 to 16 rows per CTA.  Arithmetic and exact C12 storage stay\n"
    "# unchanged; paired cold-L2 replay decides every route independently.\n",
    "# v216: repeat the 16-row retile with byte caps chosen to preserve the\n"
    "# incumbent ring depth.  This isolates grid flight count from v214's deeper\n"
    "# ring while retaining exact C12 arithmetic and cold-L2 paired gating.\n",
    1)
tail = tail.replace("v216 rowtile16", "v216 rowtile16-depth")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
