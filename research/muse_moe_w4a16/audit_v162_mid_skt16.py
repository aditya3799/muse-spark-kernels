"""Static contract audit for the corrected v162 M16 diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V139 = ROOT / "submission_v139_wgs_kperm3.py"
V161 = ROOT / "submission_v161_mid_skt16_diag.py"
V162 = ROOT / "submission_v162_mid_skt16_diag.py"
EXPECTED = {
    "v139": "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    "v161": "4832D00FCA83D81782528F3B056C2063D80BDB2207CC3BA77F1DE6E19F3AAA51",
    "active": "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def cuda_source(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("CUDA_SRC not found")


def main() -> None:
    assert hashlib.sha256(V139.read_bytes()).hexdigest().upper() == EXPECTED["v139"]
    assert hashlib.sha256(V161.read_bytes()).hexdigest().upper() == EXPECTED["v161"]
    assert hashlib.sha256((ROOT / "submission.py").read_bytes()).hexdigest().upper() == EXPECTED["active"]
    ast.parse(V162.read_text())
    assert cuda_source(V162) == cuda_source(V139)

    old = V161.read_text()
    new = V162.read_text()
    expected_new = old.replace("        assert same and rel == 0.0\n", "        assert rel <= 1.0e-5\n")
    assert new == expected_new
    assert "wins >= 14 and mean <= 0.995 and median <= 0.997" in new
    assert "DIAG = True" in new

    digest = hashlib.sha256(V162.read_bytes()).hexdigest().upper()
    print("v162 audit passed")
    print(f"candidate sha256={digest}")
    print("only change from v161: exact parity -> relative L2 <= 1e-5")


if __name__ == "__main__":
    main()
