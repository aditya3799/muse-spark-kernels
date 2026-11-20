"""Generate v196: v192's seven micro-wins plus v195's QKV/m32 RS win."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_relaxed_micro_composite_v192.py"
DONOR = ROOT / "submission_plain_qkv32_rs_tournament_v195.py"
OUTPUT = ROOT / "submission_plain_composite_v196.py"

text = BASE.read_text(encoding="utf-8")
donor = DONOR.read_text(encoding="utf-8")

old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 3: return rq_go<32, 6, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
    case 4: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
    case 5: return rq_go<32, 6, 1, 0, 0, 0, 35, 1>(p, x, out, (int)KS, (int)info);
    case 6: return rq_go<32, 5, 1, 0, 0, 0, 35, 1>(p, x, out, (int)KS, (int)info);
    case 7: return rq_go<32, 4, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v192 rq switch anchor count")
text = text.replace(old_switch, new_switch, 1)

marker = "# ---------------------------------------------------------------- v195: QKV/m32 exact RS schedules."
if donor.count(marker) != 1:
    raise RuntimeError("unexpected v195 donor marker count")
block = donor[donor.index(marker):]

text = text.rstrip() + "\n\n\n" + block.rstrip() + "\n"
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
