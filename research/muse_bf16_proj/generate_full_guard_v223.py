"""Generate v223: v222 plus a third independent all-route guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v222.py"
OUTPUT = ROOT / "submission_plain_full_guard_v223.py"

text = BASE.read_text(encoding="utf-8")
marker = "# v222: repeat the all-route numerical guard"
if text.count(marker) != 1:
    raise RuntimeError("v222 guard marker mismatch")
tail = marker + text.split(marker, 1)[1]
tail = tail.replace("v222", "v223").replace("V222", "V223")
tail = tail.replace("second independent", "third independent", 1)
tail = tail.replace("second full guard", "third full guard")
tail = tail.replace("torch.arange(29 + index, n, 89", "torch.arange(43 + index, n, 83")
tail = tail.replace(
    "cols = (1, k // 11, (4 * k) // 11, (8 * k) // 11, k - 2)",
    "cols = (3, k // 13, (5 * k) // 13, (11 * k) // 13, k - 4)")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
