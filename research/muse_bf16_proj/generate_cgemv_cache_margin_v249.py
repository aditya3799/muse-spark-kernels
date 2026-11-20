"""Generate v249: rebase exact m=1 C12 cache policies on complete v221.

v128 measured bitwise-exact adjusted-core wins on QKV, O, and especially
Pooler, but its old global 2% score gate disabled every route.  This isolate
tests those three live geometries independently with a 1% route gate and no
aggregate promotion requirement.  The appended donor is readable source only.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_cgemv_cache_v128.py"
OUTPUT = ROOT / "submission_plain_cgemv_cache_margin_v249.py"

text = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
marker = "# v128: isolate cache residency in the m=1 C12 GEMV."
if donor.count(marker) != 1:
    raise RuntimeError("v128 cache marker mismatch")
block = marker + donor.split(marker, 1)[1]

old_shapes = """_V128_SHAPES = {
    'qkv_gate': (2304, 8192),
    'o': (8192, 1024),
    'pooler': (4096, 8192),
    'unpooler': (1024, 4096),
}"""
new_shapes = """_V128_SHAPES = {
    'qkv_gate': (2304, 8192),
    'o': (8192, 1024),
    'pooler': (4096, 8192),
}"""
if block.count(old_shapes) != 1:
    raise RuntimeError("v128 shape marker mismatch")
block = block.replace(old_shapes, new_shapes, 1)
block = block.replace("_V128_REQUIRED_CORE_RATIO = 0.97",
                      "_V128_REQUIRED_CORE_RATIO = 0.99", 1)
block = block.replace("_V128_REQUIRED_SCORE_GAIN = 0.02",
                      "_V128_REQUIRED_SCORE_GAIN = 0.0", 1)
block = block.replace("okc_bf16proj_v128_cgcache_",
                      "okc_bf16proj_v249_cgcache_", 1)
block = block.replace("v128 cgemv cache", "v249 cgemv cache")

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
