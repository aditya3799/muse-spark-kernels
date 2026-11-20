"""Generate v218: constant-footprint K-warp decomposition tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V212 = ROOT / "submission_plain_cluster2_small_v212.py"
OUTPUT = ROOT / "submission_plain_warp_split_v218.py"

text = BASE.read_text(encoding="utf-8")
source = V212.read_text(encoding="utf-8")
marker = "# v212: QKV exposes 96 single-CTA row blocks"
if source.count(marker) != 1:
    raise RuntimeError("v212 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v212", "v218").replace("V212", "V218")

old_configs = '''_V218_CONFIGS = (
    ('v218_q8_cl2_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 8),
    ('v218_q16_cl2_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 16),
    ('v218_q32_cl2_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2, 32),
    ('v218_p16_cl2_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 2, 16),
    ('v218_p32_cl2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 2, 32),
)'''
new_configs = '''_V218_CONFIGS = (
    ('v218_q8_w2u4_r144', 'qkv_gate', 2304, 24, 3, 1, 2, 4, 2, 1, 8),
    ('v218_q8_w8u1_r144', 'qkv_gate', 2304, 24, 3, 1, 8, 1, 2, 1, 8),
    ('v218_q16_w2u4_r200', 'qkv_gate', 2304, 24, 3, 1, 2, 4, 2, 1, 16),
    ('v218_q16_w8u1_r200', 'qkv_gate', 2304, 24, 3, 1, 8, 1, 2, 1, 16),
    ('v218_o32_w4u1_r100', 'o', 8192, 32, 2, 2, 4, 1, 1, 1, 32),
    ('v218_p32_w4u1_r144', 'pooler', 4096, 32, 2, 2, 4, 1, 1, 1, 32),
)'''
if tail.count(old_configs) != 1:
    raise RuntimeError("v218 config marker mismatch")
tail = tail.replace(old_configs, new_configs, 1)
tail = tail.replace(
    "# v218: QKV exposes 96 single-CTA row blocks and pooler exposes 128.  These\n"
    "# cluster-2 neighbors double flight count and split K in half while preserving\n"
    "# the exact C12 representation and the existing slot-owner arithmetic.\n",
    "# v218: keep row tile, ring depth, stage bytes, and K-units fixed while\n"
    "# redistributing those units across consumer warps.  This isolates issue\n"
    "# parallelism from shared-reduction cost under paired cold-L2 replay.\n",
    1)
tail = tail.replace("v218 cluster2", "v218 warp-split")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
