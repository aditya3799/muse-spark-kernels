"""Generate v224: v223 plus a fourth independent all-route guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v223.py"
OUTPUT = ROOT / "submission_plain_full_guard_v224.py"

text = BASE.read_text(encoding="utf-8")
marker = "# v223: repeat the all-route numerical guard"
if text.count(marker) != 1:
    raise RuntimeError("v223 guard marker mismatch")
tail = marker + text.split(marker, 1)[1]
tail = tail.replace("v223", "v224").replace("V223", "V224")
tail = tail.replace("third independent", "fourth independent", 1)
tail = tail.replace("third full guard", "fourth full guard")
tail = tail.replace("torch.arange(43 + index, n, 83", "torch.arange(59 + index, n, 79")
tail = tail.replace(
    "cols = (3, k // 13, (5 * k) // 13, (11 * k) // 13, k - 4)",
    "cols = (5, k // 17, (7 * k) // 17, (14 * k) // 17, k - 6)")

text = text.rstrip() + "\n" + tail.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
