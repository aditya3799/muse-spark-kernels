"""Generate the complete nine-seed v139 M8 WSI diagnostic.

WSI deliberately changes reduction scheduling, so allow only tiny BF16-ordering
drift (1e-4 relative / worst row), still far below the challenge tolerance.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v139_wsi8_diag.py"
DST = ROOT / "submission_v139_wsi8_diag2.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = SRC.read_text()
    text = replace_once(
        text,
        '                if not torch.equal(outs[name], outs["main"]):\n'
        '                    mismatches = int(torch.count_nonzero(outs[name] != outs["main"]).item())\n'
        '                    raise RuntimeError(f"{name} is not bit-exact: mismatches={mismatches}, rel={rel:.6f}, max_row={max_row:.6f}")\n',
        '                mismatches = int(torch.count_nonzero(outs[name] != outs["main"]).item())\n'
        '                if rel > 1.0e-4 or max_row > 1.0e-4:\n'
        '                    raise RuntimeError(f"{name} output drift: mismatches={mismatches}, rel={rel:.6f}, max_row={max_row:.6f}")\n',
        "conservative numerical gate",
    )
    text = replace_once(
        text,
        "                   max_rel == 0.0 and max_row == 0.0)\n",
        "                   max_rel <= 1.0e-4 and max_row <= 1.0e-4)\n",
        "promotion numerical gate",
    )
    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
