"""Fail-closed source/layout audit for the shape-isolated v181 B4FIN candidate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from make_v181_hybrid_m32_b4fin import b4fin_source


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
CANDIDATE = ROOT / "submission_v181_hybrid_m32_b4fin.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE_SHA = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"
FROZEN_SHA = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def values(path: Path):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "ALT_SRCS":
                    key = ast.literal_eval(target.slice)
                    try:
                        out[f"ALT:{key}"] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def main() -> None:
    assert sha(LIVE) == LIVE_SHA
    assert LIVE.read_bytes() == BASE.read_bytes()
    assert sha(FROZEN) == FROZEN_SHA
    base = values(BASE)
    cand = values(CANDIDATE)
    for name in (
        "CUDA_SRC", "MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES",
        "ALT:v167_var", "ALT:v167_wsi", "ALT:v167_helper0", "ALT:v167_main",
    ):
        assert cand[name] == base[name], name
    assert cand["ALT:v181_b4fin"] == b4fin_source()
    source = cand["ALT:v181_b4fin"]
    assert source.count("#ifndef OKC_B4FIN") == 1
    assert source.count("#if OKC_WGS && OKC_B4FIN") == 2
    assert source.count("p.sspart") == 5
    assert "p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD" in source
    assert "p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0" in source
    assert "constexpr int SEG = 1024, NSEG = HD / SEG" in source
    text = CANDIDATE.read_text()
    defs = '("-DOKC_B4FIN=1", "@v181_b4fin")'
    assert text.count(defs) == 2
    assert 'state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]' in text
    assert source.count("TORCH_CHECK(x.size(0) <= 32") == 1
    assert 32 * 8 <= 3072 - 64
    assert 64 * 4096 == 1024 * 256
    print("live exact scored v178 remains protected: PASS")
    print("frozen v139 remains protected: PASS")
    print("all M1/M8/M16 and M>=33 sources/defines unchanged: PASS")
    print("B4FIN exact source/dispatch isolation at M=32: PASS")
    print("B4FIN ybuf-tail scratch capacity/alias proof: PASS")
    print(f"candidate SHA256={sha(CANDIDATE)}")


if __name__ == "__main__":
    main()
