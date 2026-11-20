"""Fail-closed source, packing, scratch, and dispatch audit for v183."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
CANDIDATE = ROOT / "submission_v183_hybrid_large_b4.py"
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s052_large_b4_partial_norm.py"
LIVE_SHA = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def values(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "ALT_SRCS":
                try:
                    out[f"ALT:{ast.literal_eval(target.slice)}"] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return out


def functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }


def literal_assignment(path: Path, name: str):
    return values(path)[name]


def main() -> None:
    assert sha(LIVE) == LIVE_SHA and LIVE.read_bytes() == BASE.read_bytes()
    base, candidate = values(BASE), values(CANDIDATE)
    for name in (
        "CUDA_SRC", "MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES",
        "V167_MAIN_DEFINES", "ALT:v167_var", "ALT:v167_wsi", "ALT:v167_helper0", "ALT:v167_main",
    ):
        assert candidate[name] == base[name], name

    exact_source = literal_assignment(DECODER, "MOE_CUDA_SRC")
    exact_defines = tuple(literal_assignment(DECODER, "MOE_DEFINES"))
    assert candidate["ALT:s052_large_b4"] == exact_source
    assert tuple(candidate["S052_MAIN_DEFINES"]) == exact_defines
    assert "#define OKC_BIGB4 0" in exact_source
    assert "large B4FIN build supports M <= 256" in exact_source

    bf, cf = functions(BASE), functions(CANDIDATE)
    for name in ("_v167_wgs_copy", "_kperm_rows", "_make_scratch", "_prepare_layout"):
        assert ast.dump(cf[name]) == ast.dump(bf[name]), name

    text = CANDIDATE.read_text()
    large_defs = '("-DOKC_B4FIN=1", "-DOKC_BIGB4=1", "@s052_large_b4")'
    assert text.count(large_defs) == 2
    assert 's052_exact = "@s052_large_b4" in defines' in text
    assert '*([] if v167_small or s052_exact else ["--use_fast_math"' in text
    assert '_v167_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), False)' in text
    assert '_v167_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, False)' in text
    assert 'state["scratch"] = _make_scratch(dev)' in text
    assert 'state["sfx"] = ("-DOKC_PRMTSC=1",)' in text
    assert 'state["large_b4"] if _M <= 256 else state["larger"]' in text
    assert 'elif M <= 256:' in text
    assert sha(CANDIDATE) != LIVE_SHA

    print("live exact scored v178 remains protected: PASS")
    print("all v178 CUDA sources, define tuples, packers, and scratch helpers preserved: PASS")
    print("large-B4 source and defines are exact decoder S052: PASS")
    print("33..256 dispatch uses private non-SC6 KPERM3 weights/scratch; M257+ stays v178: PASS")
    print(f"candidate SHA256={sha(CANDIDATE)}")


if __name__ == "__main__":
    main()
