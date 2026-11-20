"""Offline contract, reproducibility, and address audit for v166 diagnostic."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v166_wgs_epiaddr_diag.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def kperm3_col(col: int) -> int:
    return (
        (col & ~63)
        + (((col >> 1) & 3) << 4)
        + (((col >> 3) & 1) << 3)
        + (((col >> 4) & 3) << 1)
        + (col & 1)
    )


def main() -> None:
    text = SOURCE.read_text()
    ast.parse(text)
    for path, digest in EXPECTED.items():
        assert sha256(path) == digest, path.name

    assert text.count("#define OKC_WGSEPIADDR 0") == 1
    assert text.count("#if OKC_WGSEPIADDR") == 2
    assert '"epi-addr": tuple(MAIN_DEFINES) + ("-DOKC_WGSEPIADDR=1",)' in text
    assert '"poly-silu":' not in text
    assert "if not torch.equal(outs[name], outs[\"main\"]):" in text
    assert "max_rel == 0.0 and max_row == 0.0" in text
    assert "DIAG = True" in text

    # Exhaust every permuted output column, legal token-tile count, token tile,
    # and both unrolled rows for phase A.
    for col in range(4096):
        pcol = kperm3_col(col)
        for ntt in range(1, 9):
            for nt in range(ntt):
                for t in range(4):
                    base = (
                        (((pcol >> 8) * ntt + nt) * 32 + ((pcol & 255) >> 3)) * 64
                        + (2 * t) * 8
                        + (pcol & 7)
                    )
                    for h in range(2):
                        tk = nt * 8 + 2 * t + h
                        old = (
                            (((pcol >> 8) * ntt + (tk >> 3)) * 32 + ((pcol & 255) >> 3)) * 64
                            + (tk & 7) * 8
                            + (pcol & 7)
                        )
                        assert old == base + h * 8

    # Phase B's second token row is exactly HD floats after its first.
    hd = 4096
    for jst in (0, 1, 127, 2048):
        for nt in range(8):
            for t in range(4):
                tok0 = nt * 8 + 2 * t
                for n0 in (0, 7, 4095):
                    assert (jst + tok0 + 1) * hd + n0 == (jst + tok0) * hd + n0 + hd

    print("protected hashes: PASS")
    print("diagnostic contract and bit-exact gate: PASS")
    print("phase-A/phase-B address equivalence: PASS")
    print("v166 diagnostic audit: PASS")


if __name__ == "__main__":
    main()
