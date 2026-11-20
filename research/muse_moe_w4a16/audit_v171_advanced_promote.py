"""Fail-closed audit for the exact-source v171 promotion candidate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v167_keell0renz_890809c.py"
CANDIDATE = ROOT / "submission_v171_advanced_promote.py"
FROZEN = ROOT / "submission_v139_wgs_kperm3.py"
LIVE = ROOT / "submission.py"
MARKER = "# v171 promotion: exact validated v167 timed sources and dispatch\n"
EXPECTED = {
    FROZEN: "A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09",
    LIVE: "57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38",
}
EXPECTED_SOURCE_TEXT = "F3E1C2DF273D6CEAC9BDA357E03945752CF6CC02D23085D443DB22429C4DFC8B"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha_bytes(path.read_bytes())
        assert got == expected, (path.name, got, expected)
    source = SOURCE.read_text().rstrip()
    source_text_hash = sha_bytes(source.encode())
    assert source_text_hash == EXPECTED_SOURCE_TEXT, (source_text_hash, EXPECTED_SOURCE_TEXT)
    candidate = CANDIDATE.read_text()
    assert candidate == source + "\n\n" + MARKER
    ast.parse(candidate)
    print("protected frozen/live and validated source hashes: PASS")
    print("candidate differs from validated source by promotion marker only: PASS")
    print(f"candidate SHA256={sha_bytes(CANDIDATE.read_bytes())}")


if __name__ == "__main__":
    main()
