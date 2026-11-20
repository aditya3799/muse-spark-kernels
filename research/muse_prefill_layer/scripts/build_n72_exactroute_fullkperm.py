"""Enable large-T lossless K64 only for full-attention shapes."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
src = root / "submission_n72_exactroute_pin8k_candidate.py"
out = root / "submission_n72_exactroute_fullkperm_candidate.py"
text = src.read_text()

old = "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, p2=None):"
new = "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, p2=None, allow_large_kperm=True):"
assert text.count(old) == 1
text = text.replace(old, new, 1)

old = '''        mode == "fused"
        and ((KPERM_MAX_T and T < KPERM_MAX_T) or (KPERM_LARGE_MIN_T and T >= KPERM_LARGE_MIN_T))
'''
new = '''        mode == "fused"
        and ((KPERM_MAX_T and T < KPERM_MAX_T)
             or (allow_large_kperm and KPERM_LARGE_MIN_T and T >= KPERM_LARGE_MIN_T))
'''
assert text.count(old) == 1
text = text.replace(old, new, 1)

old = '    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit, p2=p2)\n'
new = ('    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit, p2=p2,\n'
       '                      allow_large_kperm=not bool(cfg["window"]))\n')
assert text.count(old) == 1
text = text.replace(old, new, 1)

for forbidden in (
    "ROUTE_SHAPING",
    "ROUTE_SHAPE_MIN_T",
    "route_shape_",
    "candsc",
    "checker-optional",
    "base64.b64decode",
    "zlib.decompress",
):
    assert forbidden not in text, forbidden
assert "UNPOOL_FP8 = False" in text and "O_FP8 = False" in text
ast.parse(text)
compile(text, str(out), "exec")
out.write_bytes(text.encode())
print(out.name, len(text.encode()))
