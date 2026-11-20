"""Generate v214: one-wave 16-row retile tournament for small dense routes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V212 = ROOT / "submission_plain_cluster2_small_v212.py"
OUTPUT = ROOT / "submission_plain_rowtile16_small_v214.py"

text = BASE.read_text(encoding="utf-8")
source = V212.read_text(encoding="utf-8")
marker = "# v212: QKV exposes 96 single-CTA row blocks"
if source.count(marker) != 1:
    raise RuntimeError("v212 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v212", "v214").replace("V212", "V214")

old_configs = '''_V214_CONFIGS = (
    ('v214_q8_cl2_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 8),
    ('v214_q16_cl2_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 16),
    ('v214_q32_cl2_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2, 32),
    ('v214_p16_cl2_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 2, 16),
    ('v214_p32_cl2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 2, 32),
)'''
new_configs = '''_V214_CONFIGS = (
    ('v214_q8_r16a_r144', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 2, 1, 8),
    ('v214_q8_r16b_r144', 'qkv_gate', 2304, 16, 1, 2, 2, 4, 2, 1, 8),
    ('v214_q16_r16a_r200', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 2, 1, 16),
    ('v214_q16_r16b_r200', 'qkv_gate', 2304, 16, 1, 2, 2, 4, 2, 1, 16),
    ('v214_p16_r16_r192', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1, 16),
    ('v214_p32_r16_r144', 'pooler', 4096, 16, 1, 2, 2, 2, 1, 1, 32),
)'''
if tail.count(old_configs) != 1:
    raise RuntimeError("v214 config marker mismatch")
tail = tail.replace(old_configs, new_configs, 1)
tail = tail.replace(
    "# v214: QKV exposes 96 single-CTA row blocks and pooler exposes 128.  These\n"
    "# cluster-2 neighbors double flight count and split K in half while preserving\n"
    "# the exact C12 representation and the existing slot-owner arithmetic.\n",
    "# v214: retile QKV from 24 to 16 rows per CTA (96 to 144 CTAs), and\n"
    "# pooler from 32 to 16 rows per CTA.  Arithmetic and exact C12 storage stay\n"
    "# unchanged; paired cold-L2 replay decides every route independently.\n",
    1)
tail = tail.replace("v214 cluster2", "v214 rowtile16")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
