"""Fail-closed audit for the exact-source v173 hybrid promotion candidate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v172_hybrid_small.py"
CANDIDATE = ROOT / "submission_v173_hybrid_promote.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
MARKER = "# v173 promotion: exact Muse-validated v172 hybrid dispatch\n"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    SOURCE: "0C8E03C70F13D04C9AC67C501CC5452D429C2F77791F84A335AC1FDFB1613D14",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha_bytes(path.read_bytes())
        assert got == expected, (path.name, got, expected)
    source = SOURCE.read_text().rstrip()
    candidate = CANDIDATE.read_text()
    assert candidate == source + "\n\n" + MARKER
    ast.parse(candidate)
    print("protected frozen/live and Muse-validated source hashes: PASS")
    print("candidate differs from v172 by promotion marker only: PASS")
    print(f"candidate SHA256={sha_bytes(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
