"""Build the exact-route/BF16-dense base with keell0renz's N72 attention."""

import ast
from pathlib import Path

root = Path(__file__).resolve().parent.parent
base = (root / "submission_scored_a272332083d04031.py").read_text()
donor = (root / "submission_keell0renz_fair_base.py").read_text()
out = root / "submission_n72_legal_hybrid_candidate.py"

start = donor.index(
    "# ---------------------------------------------------------------------------------------------- ZhimaoL fmha7"
)
end = donor.index("def kernel_fn(data, state):", start)
block = donor[start:end].rstrip() + "\n\n\n"
marker = "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, p2=None):"
assert base.count(marker) == 1
base = base.replace(marker, block + marker, 1)

assert base.count("    if SELFTEST:\n") == 1
base = base.replace("    if SELFTEST:\n", "    _zh_prepare(st)\n    if SELFTEST:\n", 1)

kpos = base.index("def kernel_fn(data, state):")
head, tail = base[:kpos], base[kpos:]
old = "    av = _attn_variant(st, cfg)\n    pre = None\n"
new = "    av = _attn_variant(st, cfg)\n    zh = _zh_ok(st, cfg)\n    pre = None\n"
assert tail.count(old) == 1
tail = tail.replace(old, new, 1)
old = "    split = av is not None and T <= PREP_SPLIT_MAX_T"
new = "    split = not zh and av is not None and T <= PREP_SPLIT_MAX_T"
assert tail.count(old) == 1
tail = tail.replace(old, new, 1)
old = "    if av is not None:\n"
new = "    if zh:\n        og = _zh_attn(st, data, q8, qkv, cfg)\n    elif av is not None:\n"
assert tail.count(old) >= 1
tail = tail.replace(old, new, 1)
base = head + tail

for required in (
    "ROUTE_SHAPING = False",
    "UNPOOL_FP8 = False",
    "O_FP8 = False",
    "POOL8_TUNE = False",
    "FP8_SMALL_TUNE = False",
):
    assert required in base, required
for forbidden in ("base64.b64decode", "zlib.decompress", "exec(compile("):
    assert forbidden not in base, forbidden
ast.parse(base)
compile(base, str(out), "exec")
out.write_bytes(base.encode())
print(out.name, len(base.encode()))
