"""Combine the exact B32 stage-5 and q8 uneven-share candidates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
text = (ROOT / "submission_sweep_stage5_current.py").read_text(encoding="utf-8")
replacements = {
    "#define OKC_W0ENABLE 0  // flat coop (q_len 8):":
        "#define OKC_W0ENABLE 1  // flat coop (q_len 8):",
    "const bool w0on = OKC_W0ENABLE && p.coop && p.resident &&":
        "const bool w0on = OKC_W0ENABLE && L == 8 && p.coop && p.resident &&",
    "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 &&":
        "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0L1 &&",
    '_build(tag="_q1occ3_l264_q8deepb32", flags=("-DOKC_CONT=1",))':
        '_build(tag="_q1occ3_l264_q8deepb32_w0", flags=("-DOKC_CONT=1",))',
}

for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"stage5/W0 anchor changed: {old!r} -> {text.count(old)}")
    text = text.replace(old, new, 1)

out = ROOT / "submission_sweep_stage5_w0_current.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)
