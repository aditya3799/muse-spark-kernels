"""Fail-closed structural audit for the clean v139 M8 WSI diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
DIAG = ROOT / "submission_v139_wsi8_diag.py"
EXPECTED = {
    FROZEN.name: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE.name: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def assignments(path: Path) -> dict[str, object]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def function_dump(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{name} not found in {path.name}")


def main() -> None:
    for name, expected in EXPECTED.items():
        got = sha(ROOT / name)
        assert got == expected, (name, got, expected)

    frozen, diag = assignments(FROZEN), assignments(DIAG)
    for name in ("MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES",
                 "TINY_MAX_M", "TINY2_MAX_M", "MID_MAX_M"):
        assert diag[name] == frozen[name], (name, diag[name], frozen[name])
    assert function_dump(FROZEN, "kernel_fn") == function_dump(DIAG, "kernel_fn")

    text = DIAG.read_text()
    assert text.count("#define OKC_WSI 0") == 1
    assert text.count("__device__ void ws_issuer") == 1
    assert text.count("-DOKC_WSI=1") == 1
    assert text.count('"wsi": tuple(TINY2_DEFINES)') == 1
    assert "shapes = (8,)" in text
    assert "range(9)" in text and "range(21)" in text
    assert "med_ratio[8] <= 0.980" in text and "wins[8] >= 7" in text
    assert "torch.equal(outs[name], outs[\"main\"])" in text
    print(f"protected frozen/live hashes: PASS")
    print(f"scored Python dispatch and define tables match frozen v139: PASS")
    print(f"WSI diagnostic structure and fail-closed gates: PASS")
    print(f"diagnostic SHA256={sha(DIAG)}")


if __name__ == "__main__":
    main()
