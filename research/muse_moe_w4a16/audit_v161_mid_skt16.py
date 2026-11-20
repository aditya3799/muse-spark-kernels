"""Static contract audit for the v161 M16 SKT16 diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v139_wgs_kperm3.py"
CANDIDATE = ROOT / "submission_v161_mid_skt16_diag.py"
BASE_SHA = "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09"
ACTIVE_SHA = "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38"


def assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name!r} not found")


def main() -> None:
    assert hashlib.sha256(BASE.read_bytes()).hexdigest().upper() == BASE_SHA
    assert hashlib.sha256((ROOT / "submission.py").read_bytes()).hexdigest().upper() == ACTIVE_SHA
    base_text = BASE.read_text()
    cand_text = CANDIDATE.read_text()
    base_tree = ast.parse(base_text)
    cand_tree = ast.parse(cand_text)

    # The candidate compiles the exact frozen CUDA and preserves official dispatch.
    assert assignment(base_tree, "CUDA_SRC") == assignment(cand_tree, "CUDA_SRC")
    assert assignment(base_tree, "MID_DEFINES") == assignment(cand_tree, "MID_DEFINES")
    assert assignment(base_tree, "MAIN_DEFINES") == assignment(cand_tree, "MAIN_DEFINES")
    assert 'elif MID_DEFINES and M <= MID_MAX_M:' in cand_text
    assert 'ext = _ext(tuple(MID_DEFINES) + sfx, use_mul=state["use_mul"])' in cand_text

    flags = assignment(cand_tree, "MID_SKT16_DEFINES")
    required = {
        "-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_SKT=16",
        "-DOKC_WS=0", "-DOKC_NACC=4", "-DOKC_SKSC2=1", "-DOKC_FTSC2=1",
    }
    assert required.issubset(set(flags))
    assert "-DOKC_WS=1" not in flags
    assert assignment(cand_tree, "DIAG") is True
    assert 'DIAG_VARIANTS = {"skt16": MID_SKT16_DEFINES}' in cand_text

    assert "seeds = tuple(1234 + 1009 * i for i in range(21))" in cand_text
    assert "assert same and rel == 0.0" in cand_text
    assert "jobs8 =" in cand_text and "jobs16 = touched" in cand_text
    assert "wins >= 14 and mean <= 0.995 and median <= 0.997" in cand_text

    digest = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest().upper()
    print("v161 audit passed")
    print(f"candidate sha256={digest}")
    print(f"protected v139={BASE_SHA}")
    print(f"protected submission.py={ACTIVE_SHA}")


if __name__ == "__main__":
    main()
