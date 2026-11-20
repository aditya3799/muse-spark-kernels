"""Generate the exact q8-only uneven per-SM share candidate from the incumbent."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
text = (ROOT / "submission.py").read_text(encoding="utf-8")
replacements = {
    "#define OKC_W0ENABLE 0  // flat coop (q_len 8):":
        "#define OKC_W0ENABLE 1  // flat coop (q_len 8):",
    "const bool w0on = OKC_W0ENABLE && p.coop && p.resident &&":
        "const bool w0on = OKC_W0ENABLE && L == 8 && p.coop && p.resident &&",
    "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 &&":
        "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0L1 &&",
    'name="okc_attn_v298_q1occ3_np128_90a"':
        'name="okc_attn_v298_q1occ3_np128_q8w0_90a"',
}

for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"q8 W0 anchor changed: {old!r} -> {text.count(old)}")
    text = text.replace(old, new, 1)

out = ROOT / "submission_sweep_q8_w0.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)
