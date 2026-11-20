"""Fail-closed audit for the final comment-only v177 measurement retry."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v177_hybrid_m32_promote.py"
CANDIDATE = ROOT / "submission_v178_hybrid_m32_retry.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
MARKER = "# v178 retry: final identical-code measurement of v177\n"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    SOURCE: "8D1C3210691E07C5DC51751F8AB33D6FD59DBF9FB4EA76DF88829B38AC6DE999",
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
    print("protected frozen/live and scored-v177 hashes: PASS")
    print("v178 differs from v177 by retry marker only: PASS")
    print(f"candidate SHA256={sha(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
