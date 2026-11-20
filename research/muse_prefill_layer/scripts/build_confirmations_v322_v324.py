"""Create executable-identical fresh-seed confirmations for the leading first-wave variants."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    (322, "scaleshare_all_confirm", "submission_n72_scaleshare_all_v301.py"),
    (323, "stripped_control_confirm", "submission_n72_stripped_control_v303.py"),
    (324, "kperm160_192_confirm", "submission_n72_kperm160_192_v305.py"),
)


def main() -> None:
    for version, label, source in CASES:
        text = (ROOT / source).read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        lines.insert(1, f"# v{version}: executable-identical fresh-seed confirmation of {source}\n")
        text = "".join(lines)
        out = ROOT / f"submission_n72_{label}_v{version}.py"
        ast.parse(text)
        compile(text, str(out), "exec")
        out.write_text(text, encoding="utf-8")
        print(out.name, hashlib.sha256(text.encode()).hexdigest().upper())


if __name__ == "__main__":
    main()
