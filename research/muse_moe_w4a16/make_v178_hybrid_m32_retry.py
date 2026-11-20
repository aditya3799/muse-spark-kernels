"""Create the one precommitted scored retry of exact v177."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v177_hybrid_m32_promote.py"
DESTINATION = ROOT / "submission_v178_hybrid_m32_retry.py"
MARKER = "# v178 retry: final identical-code measurement of v177\n"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    assert MARKER not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + MARKER)
    print(f"wrote {DESTINATION.name}")


if __name__ == "__main__":
    main()
