"""Fail-closed structural audit for the v175 three-state hybrid."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from audit_v172_hybrid_small import subscript_strings
from make_v172_hybrid_small import helper0_source, literal_assignments


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
SMALL_BASE = ROOT / "submission_v172_hybrid_small.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
CANDIDATE = ROOT / "submission_v175_hybrid_m32.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    SMALL_BASE: "0C8E03C70F13D04C9AC67C501CC5452D429C2F77791F84A335AC1FDFB1613D14",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha(path.read_bytes())
        assert got == expected, (path.name, got, expected)

    frozen = literal_assignments(FROZEN)
    advanced = literal_assignments(ADVANCED)
    candidate = literal_assignments(CANDIDATE)
    assert candidate["CUDA_SRC"] == frozen["CUDA_SRC"]
    assert candidate["MAIN_DEFINES"] == frozen["MAIN_DEFINES"]
    assert candidate["V167_MAIN_DEFINES"] == advanced["MAIN_DEFINES"]
    assert candidate["TINY_MAX_M"] == 4 and candidate["TINY2_MAX_M"] == 8 and candidate["MID_MAX_M"] == 16

    expected_tiny = tuple(advanced["TINY_DEFINES"]) + ("-DOKC_DF=0", "@v167_var")
    expected_tiny2 = tuple(advanced["TINY2_DEFINES"]) + ("-DOKC_DF=0", "@v167_wsi")
    expected_mid = tuple(advanced["MID_DEFINES"]) + ("-DOKC_WSH0=1", "@v167_helper0")
    assert candidate["TINY_DEFINES"] == expected_tiny
    assert candidate["TINY2_DEFINES"] == expected_tiny2
    assert candidate["MID_DEFINES"] == expected_mid

    var_src = advanced["VAR_SRC"]
    patch_kind, edits = advanced["PS_PATCHES"]["t2_wsi"]
    assert patch_kind == "var"
    wsi_src = var_src
    for old, new in edits:
        wsi_src = wsi_src.replace(old, new)
    sources = subscript_strings(CANDIDATE)
    assert sources["v167_var"] == var_src
    assert sources["v167_wsi"] == wsi_src
    assert sources["v167_helper0"] == helper0_source(var_src)
    assert sources["v167_main"] == advanced["CUDA_SRC"]

    af, cf = functions(ADVANCED), functions(CANDIDATE)
    for name in ("_sc6_ok", "_sc6_wgs", "_kperm_src", "_kperm_codes", "_kperm_rows"):
        assert ast.dump(cf[name]) == ast.dump(af[name]), name
    cf["_v167_wgs_copy"].name = "_wgs_copy"
    assert ast.dump(cf["_v167_wgs_copy"]) == ast.dump(af["_wgs_copy"])

    text = CANDIDATE.read_text()
    assert 'state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]' in text
    assert 'tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",)' in text
    assert 'state["w"] = list(base["w"][:12])' in text
    assert '"-DOKC_PRMTSC=1", "-DOKC_SC6W=1"' in text
    ast.parse(text)
    print("protected frozen/live/v172 hashes: PASS")
    print("v139 large and v167 small/main CUDA sources/defines: PASS")
    print("v167 SC6/KPERM packing functions preserved exactly: PASS")
    print("M<=16 / M<=32 / M>=33 three-state dispatch: PASS")
    print(f"candidate SHA256={sha(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
