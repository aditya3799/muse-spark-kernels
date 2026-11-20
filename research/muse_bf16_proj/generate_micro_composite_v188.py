"""Generate v188: guarded plain-source composite of prior micro-route wins.

The candidate keeps the scored v187 routes intact and appends the readable
v126/v134 tournament machinery.  Every leg is validated bitwise and must beat
the v187 incumbent by 2% after replay-overhead adjustment before it can enter
dispatch.  The former whole-score 2% gate is removed because it prevented
independently qualified micro-wins from composing.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
DONOR = ROOT / "submission_composite_v134.py"
OUTPUT = ROOT / "submission_plain_micro_composite_v188.py"

text = BASE.read_text(encoding="utf-8")
donor = DONOR.read_text(encoding="utf-8")

marker = "# v126: slot-owned exact-C12 correction scheduling for every dense SK2 route."
if donor.count(marker) != 1:
    raise RuntimeError("unexpected v126/v134 donor marker count")
block = donor[donor.index(marker):]

old_gate = "_V134_REQUIRED_SCORE_GAIN = 0.02"
if block.count(old_gate) != 1:
    raise RuntimeError("unexpected v134 global-gate anchor count")
block = block.replace(old_gate, "_V134_REQUIRED_SCORE_GAIN = 0.0", 1)

old_specs = """    specs = (
        ('qkv_gate', 8, 2304, 8192, 'slot'),
        ('unpooler', 8, 1024, 4096, 'slot'),
        ('unpooler', 32, 8192, 4096, 'l2'),
        ('pooler', 1, 4096, 8192, 'pool'),
    )
"""
new_specs = """    specs = (
        ('qkv_gate', 8, 2304, 8192, 'slot'),
        ('unpooler', 8, 1024, 4096, 'slot'),
        ('o', 32, 8192, 1024, 'slot'),
        ('unpooler', 32, 8192, 4096, 'l2'),
        ('pooler', 1, 4096, 8192, 'pool'),
    )
"""
if block.count(old_specs) != 1:
    raise RuntimeError("unexpected v134 route-list anchor count")
block = block.replace(old_specs, new_specs, 1)

block = block.replace(
    "# v134: validation composite of the isolated winners from v113, v126, and\n"
    "# v130.  It remains fallback-only unless their paired cold-L2 savings, measured\n"
    "# together on this H200, clear the predeclared 2% whole-score gate.",
    "# v188: revalidate the isolated v113/v126/v130 micro-route candidates\n"
    "# directly against v187.  Each route remains fallback-only unless its own\n"
    "# paired cold-L2 result clears the predeclared adjusted 2% gate.",
    1,
)

text = text.rstrip() + "\n\n\n" + block.rstrip() + "\n"
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
