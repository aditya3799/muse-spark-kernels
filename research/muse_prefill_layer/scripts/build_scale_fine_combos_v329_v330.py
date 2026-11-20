"""Combine the leading exact scale-share mask with alternate fine-tail policies."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_n72_stripped_control_v303.py"


def emit(version: int, label: str, fine_mask: int) -> None:
    text = BASE.read_text(encoding="utf-8")
    old_scale = "#define MOE_SCALE_SHFL_ENABLED(N) ((0x00 >> ((N) / 32 - 1)) & 1)  // v303: stripped_control"
    new_scale = f"#define MOE_SCALE_SHFL_ENABLED(N) ((0x38 >> ((N) / 32 - 1)) & 1)  // v{version}: {label}"
    old_fine = "#define MOE_FINE_MASK 6  // v303: stripped_control"
    new_fine = f"#define MOE_FINE_MASK {fine_mask}  // v{version}: {label}"
    assert text.count(old_scale) == 1 and text.count(old_fine) == 1
    text = text.replace(old_scale, new_scale, 1).replace(old_fine, new_fine, 1)
    lines = text.splitlines(keepends=True)
    lines.insert(1, f"# v{version}: {label}\n")
    text = "".join(lines)
    out = ROOT / f"submission_n72_{label}_v{version}.py"
    ast.parse(text)
    compile(text, str(out), "exec")
    out.write_text(text, encoding="utf-8")
    print(out.name, hashlib.sha256(text.encode()).hexdigest().upper())


def main() -> None:
    emit(329, "scaleshare128_160_192_fine192", 4)
    emit(330, "scaleshare128_160_192_fineoff", 0)


if __name__ == "__main__":
    main()
