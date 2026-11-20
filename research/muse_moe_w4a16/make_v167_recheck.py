"""Create a kernel-identical advanced-v167 payload for an independent check."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v167_recheck.py"


def main() -> None:
    text = SOURCE.read_text()
    assert text.startswith("# okc: problem=muse_moe_w4a16\n")
    marker = "# independent aggregate validation payload; timed sources and dispatch unchanged\n"
    assert marker not in text
    DESTINATION.write_text(text.rstrip() + "\n\n" + marker)
    print(f"wrote {DESTINATION.name} ({len(text)} source bytes + marker)")


if __name__ == "__main__":
    main()
