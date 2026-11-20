"""Create the one controlled scored retry of the audited v173 hybrid."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v173_hybrid_promote.py"
DESTINATION = ROOT / "submission_v174_hybrid_retry.py"
MARKER = "# v174 retry: one precommitted measurement retry of exact v173\n"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    assert MARKER not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + MARKER)
    print(f"wrote {DESTINATION.name}")


if __name__ == "__main__":
    main()
