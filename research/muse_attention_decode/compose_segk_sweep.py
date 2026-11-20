"""Generate exact q1/q8 flat-partition segment-size variants."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
base = SOURCE.read_text(encoding="utf-8")

seg8_anchor = "#define OKC_SEGK8 600   // flat q_len 8: virtual keys per request boundary in the partition"
seg1_anchor = "#define OKC_SEGK1 400   // flat q_len 1: the same (b16 90.4 -> 88.7)"

if base.count(seg8_anchor) != 1 or base.count(seg1_anchor) != 1:
    raise SystemExit("SEGK anchors changed")

variants = {
    "q1_300": (300, 600),
    "q1_500": (500, 600),
    "q8_500": (400, 500),
    "q8_700": (400, 700),
}

for tag, (seg1, seg8) in variants.items():
    text = base.replace(seg8_anchor, seg8_anchor.replace("600", str(seg8), 1), 1)
    text = text.replace(seg1_anchor, seg1_anchor.replace("400", str(seg1), 1), 1)
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_segk_{tag}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_segk_{tag}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
