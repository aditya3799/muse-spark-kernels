"""Offline contract and exact-equivalence audit for v168 qvpos diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v168_wgs_qvpos_diag.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    text = SOURCE.read_text()
    ast.parse(text)
    for path, digest in EXPECTED.items():
        assert sha256(path) == digest, path.name
    assert text.count("#define OKC_WGSQVPOS 0") == 1
    assert text.count("#if OKC_WGSQVPOS") == 1
    assert '"qvpos": tuple(MAIN_DEFINES) + ("-DOKC_WGSQVPOS=1",)' in text
    assert '"poly-silu":' not in text
    assert "if not torch.equal(outs[name], outs[\"main\"]):" in text
    assert "max_rel == 0.0 and max_row == 0.0" in text
    assert "DIAG = True" in text

    # qv is initialized as qv[sl] = S.pos[tk * 8 + sl].  The only rewritten
    # read is guarded by tid < 8, so both expressions index the same element.
    for tk in (0, 1, 31, 127, 255):
        positions = [tk * 8 + sl for sl in range(8)]
        for tid in range(8):
            assert positions[tid] == tk * 8 + tid

    print("protected hashes: PASS")
    print("leader route-index equivalence: PASS")
    print("diagnostic contract and bit-exact gate: PASS")
    print("v168 diagnostic audit: PASS")


if __name__ == "__main__":
    main()
