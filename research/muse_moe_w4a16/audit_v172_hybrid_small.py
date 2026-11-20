"""Fail-closed structural audit for the v172 small-kernel hybrid."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from make_v172_hybrid_small import helper0_source, literal_assignments


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
CANDIDATE = ROOT / "submission_v172_hybrid_small.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def subscript_strings(path: Path) -> dict[str, str]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "ALT_SRCS":
            continue
        slice_node = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
        key = ast.literal_eval(slice_node)
        out[key] = ast.literal_eval(node.value)
    return out


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha(path.read_bytes())
        assert got == expected, (path.name, got, expected)

    frozen = literal_assignments(FROZEN)
    advanced = literal_assignments(ADVANCED)
    candidate = literal_assignments(CANDIDATE)
    assert candidate["CUDA_SRC"] == frozen["CUDA_SRC"]
    assert candidate["MAIN_DEFINES"] == frozen["MAIN_DEFINES"]
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

    text = CANDIDATE.read_text()
    assert 'v167_small = any(d.startswith("@v167_") for d in defines)' in text
    assert 'state["tiny"] if data["hidden_states"].shape[0] <= MID_MAX_M else state["larger"]' in text
    assert 'if use_mul:\n        mods = [tuple(_main_defines(state["sfx"])) + state["sfx"]]' in text
    ast.parse(text)
    print("protected frozen/live hashes: PASS")
    print("v139 main CUDA/defines preserved exactly: PASS")
    print("v167 small sources/defines embedded exactly: PASS")
    print("separate _POS small-state / _POS_MUL large-state dispatch: PASS")
    print(f"candidate SHA256={sha(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
