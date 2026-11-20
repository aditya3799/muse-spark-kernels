"""Generate v192: corrected broad guarded micro-route composite on v187.

This follow-up keeps v188's two measured winners and retests every additional
slot-owner route that was within 1% adjusted core time in the same official
run.  Each route must be bitwise-correct and independently beat v187 by at
least 1% after replay-overhead adjustment; there is no aggregate gate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
DONOR = ROOT / "submission_composite_v134.py"
OUTPUT = ROOT / "submission_plain_relaxed_micro_composite_v192.py"

text = BASE.read_text(encoding="utf-8")
donor = DONOR.read_text(encoding="utf-8")
marker = "# v126: slot-owned exact-C12 correction scheduling for every dense SK2 route."
if donor.count(marker) != 1:
    raise RuntimeError("unexpected v126/v134 donor marker count")
block = donor[donor.index(marker):]

if block.count("_V134_REQUIRED_SCORE_GAIN = 0.02") != 1:
    raise RuntimeError("unexpected v134 global-gate anchor count")
block = block.replace(
    "_V134_REQUIRED_SCORE_GAIN = 0.02",
    "_V134_REQUIRED_SCORE_GAIN = 0.0",
    1,
)

old_loads = "_V134_LOAD_US = {'decode1': 34.65, 'verify1': 37.22, 'decode32': 51.69}"
new_loads = "_V134_LOAD_US = {'decode1': 34.65, 'decode16': 47.47, 'decode32': 51.69, 'verify1': 37.22}"
if block.count(old_loads) != 1:
    raise RuntimeError("unexpected v134 load-table anchor count")
block = block.replace(old_loads, new_loads, 1)

old_specs = """    specs = (
        ('qkv_gate', 8, 2304, 8192, 'slot'),
        ('unpooler', 8, 1024, 4096, 'slot'),
        ('unpooler', 32, 8192, 4096, 'l2'),
        ('pooler', 1, 4096, 8192, 'pool'),
    )
"""
new_specs = """    specs = (
        ('qkv_gate', 8, 2304, 8192, 'slot'),
        ('qkv_gate', 16, 2304, 8192, 'slot'),
        ('o', 16, 8192, 1024, 'slot'),
        ('o', 32, 8192, 1024, 'slot'),
        ('pooler', 32, 4096, 8192, 'slot'),
        ('unpooler', 16, 8192, 4096, 'slot'),
        ('unpooler', 32, 8192, 4096, 'l2'),
        ('pooler', 1, 4096, 8192, 'pool'),
    )
"""
if block.count(old_specs) != 1:
    raise RuntimeError("unexpected v134 route-list anchor count")
block = block.replace(old_specs, new_specs, 1)

if block.count("            required = 0.98") != 1:
    raise RuntimeError("unexpected v134 route-gate anchor count")
block = block.replace("            required = 0.98", "            required = 0.99", 1)

old_savings = "    savings = {'decode1': 0.0, 'verify1': 0.0, 'decode32': 0.0}"
new_savings = "    savings = {'decode1': 0.0, 'decode16': 0.0, 'decode32': 0.0, 'verify1': 0.0}"
if block.count(old_savings) != 1:
    raise RuntimeError("unexpected v134 savings-table anchor count")
block = block.replace(old_savings, new_savings, 1)

old_map = "            load = 'decode1' if m == 1 else 'verify1' if m == 8 else 'decode32'"
new_map = "            load = ('decode1' if m == 1 else 'verify1' if m == 8 else\n                    'decode16' if m == 16 else 'decode32')"
if block.count(old_map) != 1:
    raise RuntimeError("unexpected v134 load-map anchor count")
block = block.replace(old_map, new_map, 1)

block = block.replace(
    "# v134: validation composite of the isolated winners from v113, v126, and\n"
    "# v130.  It remains fallback-only unless their paired cold-L2 savings, measured\n"
    "# together on this H200, clear the predeclared 2% whole-score gate.",
    "# v192: broad per-route revalidation against v187.  Every candidate is\n"
    "# fallback-only unless its own paired cold-L2 adjusted ratio is at most 0.99.",
    1,
)

text = text.rstrip() + "\n\n\n" + block.rstrip() + "\n"
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
