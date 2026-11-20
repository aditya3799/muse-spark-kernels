"""Generate v220: unpooler/m32 row/K-warp decomposition tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V212 = ROOT / "submission_plain_cluster2_small_v212.py"
OUTPUT = ROOT / "submission_plain_unpool32_decomp_v220.py"

text = BASE.read_text(encoding="utf-8")
source = V212.read_text(encoding="utf-8")
marker = "# v212: QKV exposes 96 single-CTA row blocks"
if source.count(marker) != 1:
    raise RuntimeError("v212 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v212", "v220").replace("V212", "V220")

old_configs = '''_V220_CONFIGS = (
    ('v220_q8_cl2_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 8),
    ('v220_q16_cl2_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 16),
    ('v220_q32_cl2_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2, 32),
    ('v220_p16_cl2_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 2, 16),
    ('v220_p32_cl2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 2, 32),
)'''
new_configs = '''_V220_CONFIGS = (
    ('v220_u32_w1u4_r144', 'unpooler', 8192, 64, 4, 2, 1, 4, 1, 1, 32),
    ('v220_u32_w4u1_r144', 'unpooler', 8192, 64, 4, 2, 4, 1, 1, 1, 32),
    ('v220_u32_nwr1_r144', 'unpooler', 8192, 64, 8, 1, 2, 2, 1, 1, 32),
    ('v220_u32_nwr4_r144', 'unpooler', 8192, 64, 2, 4, 2, 2, 1, 1, 32),
    ('v220_u32_nwr8_r144', 'unpooler', 8192, 64, 1, 8, 2, 2, 1, 1, 32),
)'''
if tail.count(old_configs) != 1:
    raise RuntimeError("v220 config marker mismatch")
tail = tail.replace(old_configs, new_configs, 1)
tail = tail.replace(
    "# v220: QKV exposes 96 single-CTA row blocks and pooler exposes 128.  These\n"
    "# cluster-2 neighbors double flight count and split K in half while preserving\n"
    "# the exact C12 representation and the existing slot-owner arithmetic.\n",
    "# v220: keep unpooler/m32's 64-row tile, grid, stage bytes, and depth fixed\n"
    "# while redistributing K units and output rows across warps.  Each ordinary\n"
    "# slot-owner candidate is paired against v210's current exact L2 route.\n",
    1)
tail = tail.replace("v220 cluster2", "v220 unpool32-decomp")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
