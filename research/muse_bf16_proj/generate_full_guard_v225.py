"""Generate v225: v224 plus a fifth independent all-route guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v224.py"
OUTPUT = ROOT / "submission_plain_full_guard_v225.py"

text = BASE.read_text(encoding="utf-8")
marker = "# v224: repeat the all-route numerical guard"
if text.count(marker) != 1:
    raise RuntimeError("v224 guard marker mismatch")
tail = marker + text.split(marker, 1)[1]
tail = tail.replace("v224", "v225").replace("V224", "V225")
tail = tail.replace("fourth independent", "fifth independent", 1)
tail = tail.replace("fourth full guard", "fifth full guard")
tail = tail.replace("torch.arange(59 + index, n, 79", "torch.arange(71 + index, n, 73")
tail = tail.replace(
    "cols = (5, k // 17, (7 * k) // 17, (14 * k) // 17, k - 6)",
    "cols = (7, k // 19, (8 * k) // 19, (16 * k) // 19, k - 8)")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
