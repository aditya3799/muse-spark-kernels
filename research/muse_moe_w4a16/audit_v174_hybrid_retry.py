"""Fail-closed audit for the single comment-only v174 scoring retry."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v173_hybrid_promote.py"
CANDIDATE = ROOT / "submission_v174_hybrid_retry.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
MARKER = "# v174 retry: one precommitted measurement retry of exact v173\n"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
    SOURCE: "E0F6F1AED5B90422D1FA7CE0BF6642D52199311D0EEB4EB6F8AD49370C992385",
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
    print("protected frozen/live and scored-v173 hashes: PASS")
    print("v174 differs from v173 by retry marker only: PASS")
    print(f"candidate SHA256={sha_bytes(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
