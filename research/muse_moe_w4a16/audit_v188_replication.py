"""Prove that v188 differs from v187 only by a leading comment."""

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).parent
BASE = ROOT / "submission_v187_combined_b4_large.py"
CANDIDATE = ROOT / "submission_v188_combined_replication.py"
LIVE = ROOT / "submission.py"
BASE_SHA = "9dd8a95d23509c4194bff87c09a0cb0b82b99356f1cffc4ebc97c134c07af3d5"
LIVE_SHA = "b462ebd989eb665f2070828dafa5fe13a80e82ea328963bc543d63615e268c94"
NONCE = "# v188: behavior-identical scored replication of audited v187\n"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = BASE.read_text()
    candidate = CANDIDATE.read_text()
    assert sha(BASE) == BASE_SHA
    assert sha(LIVE) == LIVE_SHA
    first, rest = base.split("\n", 1)
    assert candidate == first + "\n" + NONCE + rest
    assert ast.dump(ast.parse(candidate), include_attributes=False) == ast.dump(ast.parse(base), include_attributes=False)
    print(f"live={LIVE_SHA}")
    print(f"v187={BASE_SHA}")
    print(f"v188={sha(CANDIDATE)}")
    print("v188 AST-identical replication audit: PASS")


if __name__ == "__main__":
    main()
