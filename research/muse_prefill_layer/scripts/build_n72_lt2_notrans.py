from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_n72_exactroute_smallkperm_pin8k_lt2fixed_candidate.py"
TARGET = ROOT / "submission_n72_exactroute_smallkperm_pin8k_lt2notrans_candidate.py"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("LT_PIN2 = {")
    end = text.index("LT_ALGO_TOL", start)
    replacement = '''LT_PIN2 = {
    # Keep only layouts that reuse the canonical weight orientation.  The
    # transposed T=753 pooler/unpooler copies save a few microseconds locally
    # but enlarge the persistent weight set and can disturb L2 residency.
    (753, "o"): [(0, [66, 495, 1, 0, 0, 7, 35, 0, 5])],
    (3012, "o"): [(0, [66, 189, 1, 0, 0, 2, 35, 0, 5]), (0, [66, 197, 1, 0, 0, 2, 35, 0, 5])],
    (3012, "qkv"): [(0, [66, 496, 1, 0, 0, 5, 35, 0, 2])],
    (16384, "o"): [(1, [66, 29, 1, 0, 0, 1, 35, 0, 3]), (0, [66, 29, 1, 0, 0, 1, 35, 0, 3])],
}
'''
    out = text[:start] + replacement + text[end:]
    ast.parse(out)
    compile(out, str(TARGET), "exec")
    assert "(753, \"pooler\")" not in replacement
    assert "(753, \"unpooler\")" not in replacement
    assert "UNPOOL_FP8 = False" in out
    assert "O_FP8 = False" in out
    assert "POOL8_TUNE = False" in out
    assert "FP8_SMALL_TUNE = False" in out
    for forbidden in ("ROUTE_SHAPING", "shape_routes", "route_shape_", "candsc"):
        assert forbidden not in out
    TARGET.write_text(out, encoding="utf-8")
    print(TARGET)
    print(hashlib.sha256(out.encode()).hexdigest().upper())


if __name__ == "__main__":
    main()
