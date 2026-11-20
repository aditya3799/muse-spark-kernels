"""Create a content-distinct, behavior-identical final replication of v187."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v187_combined_b4_large.py")
OUT = Path(__file__).with_name("submission_v188_combined_replication.py")


def main():
    source = BASE.read_text()
    first, rest = source.split("\n", 1)
    OUT.write_text(first + "\n# v188: behavior-identical scored replication of audited v187\n" + rest)
    print(OUT)


if __name__ == "__main__":
    main()
