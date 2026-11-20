"""Freeze the validated advanced-v167 source as the v171 scoring candidate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v171_advanced_promote.py"
MARKER = "# v171 promotion: exact validated v167 timed sources and dispatch\n"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    assert MARKER not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + MARKER)
    print(f"wrote {DESTINATION.name}")


if __name__ == "__main__":
    main()
