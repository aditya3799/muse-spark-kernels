from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_n72_exactroute_smallkperm_pin8k_lt2fixed_candidate.py"
TARGET = ROOT / "submission_n72_exactroute_slidingkperm_pin8k_lt2fixed_candidate.py"


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new, 1)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, p2=None):",
        "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None, p2=None, allow_large_kperm=True):",
    )
    text = replace_once(
        text,
        '''        mode == "fused"
        and ((KPERM_MAX_T and T < KPERM_MAX_T) or (KPERM_LARGE_MIN_T and T >= KPERM_LARGE_MIN_T))
''',
        '''        mode == "fused"
        and ((KPERM_MAX_T and T < KPERM_MAX_T)
             or (allow_large_kperm and KPERM_LARGE_MIN_T and T >= KPERM_LARGE_MIN_T))
''',
    )
    text = replace_once(
        text,
        '    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit, p2=p2)\n',
        '    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit, p2=p2,\n'
        '                      allow_large_kperm=bool(cfg["window"]))\n',
    )
    text = replace_once(
        text,
        "KPERM_LARGE_MIN_T = 0  # use lossless K64 only at T753; retain the standard large-T layout",
        "KPERM_LARGE_MIN_T = 8192  # isolate lossless K64 to the 138x-weighted 16K sliding workload",
    )
    ast.parse(text)
    compile(text, str(TARGET), "exec")
    assert 'allow_large_kperm=bool(cfg["window"])' in text
    assert "UNPOOL_FP8 = False" in text
    assert "O_FP8 = False" in text
    assert "POOL8_TUNE = False" in text
    assert "FP8_SMALL_TUNE = False" in text
    for forbidden in ("ROUTE_SHAPING", "shape_routes", "route_shape_", "candsc"):
        assert forbidden not in text
    TARGET.write_text(text, encoding="utf-8")
    print(TARGET)
    print(hashlib.sha256(text.encode()).hexdigest().upper())


if __name__ == "__main__":
    main()
