"""Freeze the Muse-validated v172 hybrid as the v173 scoring candidate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v172_hybrid_small.py"
DESTINATION = ROOT / "submission_v173_hybrid_promote.py"
MARKER = "# v173 promotion: exact Muse-validated v172 hybrid dispatch\n"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    assert MARKER not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + MARKER)
    print(f"wrote {DESTINATION.name}")


if __name__ == "__main__":
    main()
