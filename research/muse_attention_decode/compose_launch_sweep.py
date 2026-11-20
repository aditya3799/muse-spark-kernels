"""Generate exact launch-geometry variants from the active scored winner."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
text = SOURCE.read_text(encoding="utf-8")

anchor = "        ft = sms if (L == 8 and OKC_F2) else ((3 * sms) if L == 1 else target)"
if text.count(anchor) != 1:
    raise SystemExit(f"launch anchor count is {text.count(anchor)}")

variants = {
    "submission_sweep_q1ft384.py":
        "        ft = sms if (L == 8 and OKC_F2) else (((3 * sms - 12) if L == 1 else target))",
    "submission_sweep_q1ft400.py":
        "        ft = sms if (L == 8 and OKC_F2) else (((3 * sms + 4) if L == 1 else target))",
    "submission_sweep_q8ft252.py":
        "        ft = sms if (L == 8 and OKC_F2) else ((3 * sms) if L == 1 else ((target * 21) // 22))",
    "submission_sweep_q8ft276.py":
        "        ft = sms if (L == 8 and OKC_F2) else ((3 * sms) if L == 1 else (target + 12))",
}

for name, replacement in variants.items():
    out = ROOT / name
    out.write_text(text.replace(anchor, replacement, 1), encoding="utf-8", newline="\n")
    print(out)
