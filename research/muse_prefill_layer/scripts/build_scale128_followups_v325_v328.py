"""Generate fresh confirmation and 128-row scale-sharing decompositions."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_n72_stripped_control_v303.py"


def emit(version: int, label: str, mask: int) -> None:
    text = BASE.read_text(encoding="utf-8")
    old = "#define MOE_SCALE_SHFL_ENABLED(N) ((0x00 >> ((N) / 32 - 1)) & 1)  // v303: stripped_control"
    new = f"#define MOE_SCALE_SHFL_ENABLED(N) ((0x{mask:02x} >> ((N) / 32 - 1)) & 1)  // v{version}: {label}"
    assert text.count(old) == 1
    text = text.replace(old, new, 1)
    lines = text.splitlines(keepends=True)
    lines.insert(1, f"# v{version}: legal exact scale-sharing follow-up ({label})\n")
    text = "".join(lines)
    out = ROOT / f"submission_n72_{label}_v{version}.py"
    ast.parse(text)
    compile(text, str(out), "exec")
    out.write_text(text, encoding="utf-8")
    print(out.name, hashlib.sha256(text.encode()).hexdigest().upper())


def main() -> None:
    emit(325, "scaleshare128_160_192_confirm", 0x38)
    emit(326, "scaleshare128_only", 0x08)
    emit(327, "scaleshare128_192", 0x28)
    emit(328, "scaleshare128_160", 0x18)


if __name__ == "__main__":
    main()
