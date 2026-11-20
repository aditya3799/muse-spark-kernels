"""Generate v226 and v227 with fresh independent all-route guards."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def append_guard(base_name, output_name, old_version, new_version,
                 old_ordinal, new_ordinal, old_seed, new_seed,
                 old_rows, new_rows, old_cols, new_cols):
    base = ROOT / base_name
    output = ROOT / output_name
    text = base.read_text(encoding="utf-8")
    marker = f"# {old_version}: repeat the all-route numerical guard"
    if text.count(marker) != 1:
        raise RuntimeError(f"{old_version} guard marker mismatch")
    tail = marker + text.split(marker, 1)[1]
    tail = tail.replace(old_version, new_version).replace(
        old_version.upper(), new_version.upper())
    tail = tail.replace(f"{old_ordinal} independent",
                        f"{new_ordinal} independent", 1)
    tail = tail.replace(f"{old_ordinal} full guard",
                        f"{new_ordinal} full guard")
    tail = tail.replace(old_seed, new_seed)
    tail = tail.replace(old_rows, new_rows)
    tail = tail.replace(old_cols, new_cols)
    text = text.rstrip() + "\n" + tail.lstrip("\n")
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output} ({output.stat().st_size} bytes)")


append_guard(
    "submission_plain_full_guard_v225.py",
    "submission_plain_full_guard_v226.py",
    "v225", "v226", "fifth", "sixth",
    "manual_seed(222000 + index)", "manual_seed(226000 + index)",
    "torch.arange(71 + index, n, 73", "torch.arange(23 + index, n, 83",
    "cols = (7, k // 19, (8 * k) // 19, (16 * k) // 19, k - 8)",
    "cols = (9, k // 23, (11 * k) // 23, (19 * k) // 23, k - 10)",
)

append_guard(
    "submission_plain_full_guard_v226.py",
    "submission_plain_full_guard_v227.py",
    "v226", "v227", "sixth", "seventh",
    "manual_seed(226000 + index)", "manual_seed(227000 + index)",
    "torch.arange(23 + index, n, 83", "torch.arange(41 + index, n, 89",
    "cols = (9, k // 23, (11 * k) // 23, (19 * k) // 23, k - 10)",
    "cols = (11, k // 29, (13 * k) // 29, (23 * k) // 29, k - 12)",
)
