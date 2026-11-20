"""Generate v251: compose v248 pair-planes with v249 cache policies.

The v249 block captures v248 as its incumbent.  Its O/m=1 cache candidate must
therefore beat the already-selected pair-plane O kernel, while QKV and Pooler
remain independent and v248's Unpooler route is preserved.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
DONOR = ROOT / "submission_cgemv_cache_v128.py"
OUTPUT = ROOT / "submission_plain_pairplanes_cache_composite_v251.py"

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
                      "okc_bf16proj_v251_cgcache_", 1)
block = block.replace("v128 cgemv cache", "v251 cgemv cache")

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
