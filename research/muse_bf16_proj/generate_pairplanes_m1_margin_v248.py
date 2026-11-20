"""Generate v248: rebase exact C12 pair-planes for O/Unpooler m=1 on v221.

The v106 routes were bitwise exact and measured about 2.2-2.3% faster after
replay correction, just below their old 3% promotion gate.  This isolate keeps
only the two historical winners and requires a fresh 1% paired win.  The donor
block is literal source; none of v106's encoded base packaging is copied.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_c12_pairplanes_v106.py"
OUTPUT = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"

text = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
marker = "# v106: exact C12 row-paired code planes for the scored m=1 dense routes."
if donor.count(marker) != 1:
    raise RuntimeError("v106 pair-plane marker mismatch")
block = marker + donor.split(marker, 1)[1]

old_shapes = """_V106_CG_SHAPES = {
    'qkv_gate': (2304, 8192),
    'o': (8192, 1024),
    'pooler': (4096, 8192),
    'unpooler': (1024, 4096),
}"""
new_shapes = """_V106_CG_SHAPES = {
    'o': (8192, 1024),
    'unpooler': (1024, 4096),
}"""
if block.count(old_shapes) != 1:
    raise RuntimeError("v106 shape marker mismatch")
block = block.replace(old_shapes, new_shapes, 1)
block = block.replace("_V106_REQUIRED_CORE_RATIO = 0.97",
                      "_V106_REQUIRED_CORE_RATIO = 0.99", 1)
block = block.replace("okc_bf16proj_v106_c12pair_",
                      "okc_bf16proj_v248_c12pair_", 1)
block = block.replace("v106 c12 pairplanes", "v248 c12 pairplanes")
block = block.replace("v106 c12 pairplanes unavailable",
                      "v248 c12 pairplanes unavailable")

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
