"""Fail-closed audit for v177's comment-only scoring promotion."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v176_hybrid_m32_scratch.py"
CANDIDATE = ROOT / "submission_v177_hybrid_m32_promote.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
MARKER = "# v177 promotion: exact Muse-validated v176 three-state hybrid\n"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    SOURCE: "9D14D6BAA237CC1B2299622920C2EDCBEC11D8C9DF2B2DC871F3142DDBE5A0F2",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha(path.read_bytes())
        assert got == expected, (path.name, got, expected)
    source = SOURCE.read_text().rstrip()
    candidate = CANDIDATE.read_text()
    assert candidate == source + "\n\n" + MARKER
    ast.parse(candidate)
    print("protected frozen/live and Muse-validated v176 hashes: PASS")
    print("v177 differs from v176 by promotion marker only: PASS")
    print(f"candidate SHA256={sha(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
