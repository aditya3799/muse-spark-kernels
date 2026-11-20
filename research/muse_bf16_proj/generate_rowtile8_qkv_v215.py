"""Generate v215: two-wave 8-row QKV retile tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V214 = ROOT / "submission_plain_rowtile16_small_v214.py"
OUTPUT = ROOT / "submission_plain_rowtile8_qkv_v215.py"

text = BASE.read_text(encoding="utf-8")
source = V214.read_text(encoding="utf-8")
marker = "# v214: retile QKV from 24 to 16 rows per CTA"
if source.count(marker) != 1:
    raise RuntimeError("v214 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v214", "v215").replace("V214", "V215")

old_configs = '''_V215_CONFIGS = (
    ('v215_q8_r16a_r144', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 2, 1, 8),
    ('v215_q8_r16b_r144', 'qkv_gate', 2304, 16, 1, 2, 2, 4, 2, 1, 8),
    ('v215_q16_r16a_r200', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 2, 1, 16),
    ('v215_q16_r16b_r200', 'qkv_gate', 2304, 16, 1, 2, 2, 4, 2, 1, 16),
    ('v215_p16_r16_r192', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1, 16),
    ('v215_p32_r16_r144', 'pooler', 4096, 16, 1, 2, 2, 2, 1, 1, 32),
)'''
new_configs = '''_V215_CONFIGS = (
    ('v215_q8_r8a_r144', 'qkv_gate', 2304, 8, 1, 1, 4, 2, 2, 1, 8),
    ('v215_q8_r8b_r144', 'qkv_gate', 2304, 8, 1, 1, 2, 4, 2, 1, 8),
    ('v215_q16_r8a_r200', 'qkv_gate', 2304, 8, 1, 1, 4, 2, 2, 1, 16),
    ('v215_q16_r8b_r200', 'qkv_gate', 2304, 8, 1, 1, 2, 4, 2, 1, 16),
)'''
if tail.count(old_configs) != 1:
    raise RuntimeError("v215 config marker mismatch")
tail = tail.replace(old_configs, new_configs, 1)
tail = tail.replace(
    "# v215: retile QKV from 24 to 16 rows per CTA (96 to 144 CTAs), and\n"
    "# pooler from 32 to 16 rows per CTA.  Arithmetic and exact C12 storage stay\n"
    "# unchanged; paired cold-L2 replay decides every route independently.\n",
    "# v215: retile QKV from 24 to 8 rows per CTA (96 to 288 CTAs).  Test\n"
    "# both four-K-warp and two-K-warp decompositions; arithmetic and exact C12\n"
    "# storage stay unchanged and paired cold-L2 replay gates every route.\n",
    1)
tail = tail.replace("v215 rowtile16", "v215 rowtile8")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
