"""Generate v191: isolated four-stage router RK ring experiment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
OUTPUT = ROOT / "submission_plain_router_ring4_v191.py"

text = BASE.read_text(encoding="utf-8")
old = "#define RK_QS 6"
new = "#define RK_QS 4"
if text.count(old) != 1:
    raise RuntimeError("unexpected RK_QS anchor count")
text = text.replace(old, new, 1)

text = text.replace(
    "// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB:",
    "// + 1 producer warp (TMA, four-stage RK_QS ring of 24 KB:",
    1,
)

with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
