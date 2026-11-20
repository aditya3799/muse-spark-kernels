"""Generate controlled legal MoE tile-layout variants from the v300 source."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_n72_tilekperm_scaleshare_v300.py"


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def build(version: int, label: str, kperm_mask: int, scale_mask: int, enable_kperm: bool = True,
          fine_mask: int = 6) -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "#define MOE_KPERM_TILE_MASK 0x10  // use the transposed-K path only for measured-best 160-row tiles",
        f"#define MOE_KPERM_TILE_MASK 0x{kperm_mask:02x}  // v{version}: {label}",
    )
    text = replace_once(
        text,
        "#define MOE_SCALE_SHFL_ENABLED(N) 1  // exact subgroup scale sharing on every standard-layout large-T tile",
        f"#define MOE_SCALE_SHFL_ENABLED(N) ((0x{scale_mask:02x} >> ((N) / 32 - 1)) & 1)  // v{version}: {label}",
    )
    text = replace_once(
        text,
        "#define MOE_FINE_MASK 6  // measured best: 160 + 192 fine tails",
        f"#define MOE_FINE_MASK {fine_mask}  // v{version}: {label}",
    )
    if not enable_kperm:
        text = replace_once(
            text,
            "KPERM_TILE_MIN_T = 8192  # hybrid capability retained; current large-T dispatch uses standard layout plus scale sharing",
            f"KPERM_TILE_MIN_T = 0  # v{version}: tile-selective K64 disabled ({label})",
        )
    out = ROOT / f"submission_n72_{label}_v{version}.py"
    ast.parse(text)
    compile(text, str(out), "exec")
    assert "UNPOOL_FP8" not in text
    assert "O_FP8" not in text
    assert "ROUTE_SHAPING" not in text
    out.write_text(text, encoding="utf-8")
    print(out.name, len(text.encode()), hashlib.sha256(text.encode()).hexdigest().upper())


def main() -> None:
    # mask bit mapping: 32/64/96/128/160/192/224/256 rows -> bits 0..7.
    cases = [
        (301, "scaleshare_all", 0x00, 0xFF, False, 6),
        (302, "kperm160_only", 0x10, 0x00, True, 6),
        (303, "stripped_control", 0x00, 0x00, False, 6),
        (304, "kperm192_only", 0x20, 0x00, True, 6),
        (305, "kperm160_192", 0x30, 0x00, True, 6),
        (306, "kperm128_only", 0x08, 0x00, True, 6),
        (307, "kperm128_160", 0x18, 0x00, True, 6),
        (308, "kperm224_only", 0x40, 0x00, True, 6),
        (309, "kperm256_only", 0x80, 0x00, True, 6),
        (310, "kperm128_160_192", 0x38, 0x00, True, 6),
        (311, "scaleshare160_only", 0x00, 0x10, False, 6),
        (312, "scaleshare192_only", 0x00, 0x20, False, 6),
        (313, "scaleshare160_192", 0x00, 0x30, False, 6),
        (314, "scaleshare128_160_192", 0x00, 0x38, False, 6),
        (315, "scaleshare256_only", 0x00, 0x80, False, 6),
        (316, "kperm160_scaleshare160_192", 0x10, 0x30, True, 6),
        (317, "kperm160_192_scaleshare_all", 0x30, 0xFF, True, 6),
        (318, "kperm160_192_scaleshare160_192", 0x30, 0x30, True, 6),
        (319, "fine160_only_control", 0x00, 0x00, False, 2),
        (320, "fine192_only_control", 0x00, 0x00, False, 4),
        (321, "fine_off_control", 0x00, 0x00, False, 0),
    ]
    for args in cases:
        build(*args)


if __name__ == "__main__":
    main()
