"""Fail-closed audit for v182 production plus its diagnostic appendix."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from make_diag_v182_b4_pair import APPEND


ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "submission.py"
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
PROD = ROOT / "submission_v182_hybrid_m32_decoder_b4.py"
DIAG = ROOT / "submission_diag_v182_b4_pair.py"
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s082_keell0renz_latest.py"
LIVE_SHA = "B462EBD989EB665F2070828DAFA5FE13A80E82EA328963BC543D63615E268C94"


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
                    try:
                        out[f"ALT:{ast.literal_eval(target.slice)}"] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def assignment(path: Path, name: str):
    return values(path)[name]


def main() -> None:
    assert sha(LIVE) == LIVE_SHA and LIVE.read_bytes() == BASE.read_bytes()
    base, prod = values(BASE), values(PROD)
    for name in (
        "CUDA_SRC", "MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES",
        "ALT:v167_var", "ALT:v167_wsi", "ALT:v167_helper0", "ALT:v167_main",
    ):
        assert prod[name] == base[name], name
    exact = assignment(DECODER, "MOE_CUDA_SRC")
    assert prod["ALT:v182_decoder_b4"] == exact
    assert "constexpr int HD = 4096, ID = 4096" in exact
    assert "#if OKC_WGS && OKC_B4FIN" in exact
    assert "wgs_ranges(p, S);  // all 32 lanes" in exact
    text = PROD.read_text()
    defs = '("-DOKC_B4FIN=1", "@v182_decoder_b4")'
    assert text.count(defs) == 2
    assert 'state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]' in text
    assert PROD.read_bytes().rstrip(b"\r\n") + APPEND.encode("utf-8") == DIAG.read_bytes()
    diag_text = DIAG.read_text()
    assert "_B4_SEEDS = 7" in diag_text and "_B4_REPS = 24" in diag_text
    assert '"control": tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",)' in diag_text
    assert '"b4": tuple(V167_MAIN_DEFINES) + sfx + ("-DOKC_B4FIN=1", "@v182_decoder_b4")' in diag_text
    assert "scratches = {name: _v167_make_scratch(dev)" in diag_text
    assert ".okc_attrs()" not in diag_text
    print("live exact scored v178 remains protected: PASS")
    print("all non-M32 source/dispatch branches remain exact v178: PASS")
    print("M32 source is exact decoder S052 B4FIN source: PASS")
    print("seven-seed paired diagnostic uses private v167 scratch: PASS")
    print(f"production SHA256={sha(PROD)}")
    print(f"diagnostic SHA256={sha(DIAG)}")


if __name__ == "__main__":
    main()
