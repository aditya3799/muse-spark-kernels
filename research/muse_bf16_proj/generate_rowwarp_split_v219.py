"""Generate v219: constant-footprint output-row warp decomposition sweep."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
V212 = ROOT / "submission_plain_cluster2_small_v212.py"
OUTPUT = ROOT / "submission_plain_rowwarp_split_v219.py"

text = BASE.read_text(encoding="utf-8")
source = V212.read_text(encoding="utf-8")
marker = "# v212: QKV exposes 96 single-CTA row blocks"
if source.count(marker) != 1:
    raise RuntimeError("v212 tournament marker mismatch")
tail = marker + source.split(marker, 1)[1]
tail = tail.replace("v212", "v219").replace("V212", "V219")

old_configs = '''_V219_CONFIGS = (
    ('v219_q8_cl2_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 8),
    ('v219_q16_cl2_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 16),
    ('v219_q32_cl2_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2, 32),
    ('v219_p16_cl2_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 2, 16),
    ('v219_p32_cl2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 2, 32),
)'''
new_configs = '''_V219_CONFIGS = (
    ('v219_q8_nwr3w2_r144', 'qkv_gate', 2304, 24, 1, 3, 2, 4, 2, 1, 8),
    ('v219_q8_nwr3w4_r144', 'qkv_gate', 2304, 24, 1, 3, 4, 2, 2, 1, 8),
    ('v219_q16_nwr3w2_r200', 'qkv_gate', 2304, 24, 1, 3, 2, 4, 2, 1, 16),
    ('v219_q16_nwr3w4_r200', 'qkv_gate', 2304, 24, 1, 3, 4, 2, 2, 1, 16),
    ('v219_o32_nwr1_r100', 'o', 8192, 32, 4, 1, 2, 2, 1, 1, 32),
    ('v219_o32_nwr4_r100', 'o', 8192, 32, 1, 4, 2, 2, 1, 1, 32),
    ('v219_p32_nwr1_r144', 'pooler', 4096, 32, 4, 1, 2, 2, 1, 1, 32),
    ('v219_p32_nwr4_r144', 'pooler', 4096, 32, 1, 4, 2, 2, 1, 1, 32),
)'''
if tail.count(old_configs) != 1:
    raise RuntimeError("v219 config marker mismatch")
tail = tail.replace(old_configs, new_configs, 1)
tail = tail.replace(
    "# v219: QKV exposes 96 single-CTA row blocks and pooler exposes 128.  These\n"
    "# cluster-2 neighbors double flight count and split K in half while preserving\n"
    "# the exact C12 representation and the existing slot-owner arithmetic.\n",
    "# v219: keep row tile, grid, ring depth, and bytes fixed while moving output\n"
    "# tiles between NT-per-warp and NWR row-warps.  Paired cold-L2 replay isolates\n"
    "# register/issue parallelism from row-reduction overhead.\n",
    1)
tail = tail.replace("v219 cluster2", "v219 rowwarp-split")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
