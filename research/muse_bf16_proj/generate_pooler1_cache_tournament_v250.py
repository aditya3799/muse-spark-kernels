"""Generate v250: rebase the exact Pooler/m=1 cache-policy sweep on v221.

The original v130 run found policy 6 bitwise/probe exact at 16.640 versus
17.184 us, but its global 2% score gate rejected the route.  v250 keeps all
seven readable policy variants, compares them directly with complete v221,
and promotes only a valid paired win with at least a one-percent core margin.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_cgemv_cache_margin_v249.py"
DONOR = ROOT / "submission_cgemv_policy_v130.py"
OUTPUT = ROOT / "submission_plain_pooler1_cache_tournament_v250.py"

text = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
marker = "# v130: decompose v128's pooler win into L1/L2 policies."
if donor.count(marker) != 1:
    raise RuntimeError("v130 policy marker mismatch")
block = marker + donor.split(marker, 1)[1]

replacements = {
    "_V130_REQUIRED_CORE_RATIO = 0.97":
        "_V130_REQUIRED_CORE_RATIO = 0.99",
    "_V130_REQUIRED_SCORE_GAIN = 0.02":
        "_V130_REQUIRED_SCORE_GAIN = 0.0",
    "okc_bf16proj_v130_cgpolicy_":
        "okc_bf16proj_v250_cgpolicy_",
    "v130 pooler cgemv": "v250 pooler cgemv",
}
for old, new in replacements.items():
    if block.count(old) < 1:
        raise RuntimeError("v130 replacement marker mismatch: " + old)
    block = block.replace(old, new)

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
