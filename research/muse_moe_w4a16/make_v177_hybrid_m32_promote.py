"""Freeze the Muse-validated v176 three-state hybrid for scoring."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v176_hybrid_m32_scratch.py"
DESTINATION = ROOT / "submission_v177_hybrid_m32_promote.py"
MARKER = "# v177 promotion: exact Muse-validated v176 three-state hybrid\n"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    assert MARKER not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + MARKER)
    print(f"wrote {DESTINATION.name}")


if __name__ == "__main__":
    main()
