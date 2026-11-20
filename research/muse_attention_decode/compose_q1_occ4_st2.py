"""Generate exact q1 four-CTA/two-stage flat topology candidates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
text = (ROOT / "submission.py").read_text(encoding="utf-8")
replacements = {
    "#define OKC_WSMINB1 3   // q_len 1 flat:": "#define OKC_WSMINB1 4   // q_len 1 flat:",
    "#define OKC_ST1F 3      // q_len 1 flat kernel:": "#define OKC_ST1F 2      // q_len 1 flat kernel:",
    "((3 * sms) if L == 1 else target)": "((4 * sms) if L == 1 else target)",
}

for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"q1 occ4 anchor changed: {old!r} -> {text.count(old)}")
    text = text.replace(old, new, 1)

text = text.replace(
    'name="okc_attn_v298_q1occ3_np128_90a"',
    'name="okc_attn_v298_q1occ4_st2_np128_90a"',
    1,
)
out = ROOT / "submission_sweep_q1_occ4_st2.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)

np64 = text.replace(
    "#define OKC_NP 128      // producer threads",
    "#define OKC_NP 64       // producer threads",
    1,
).replace(
    'name="okc_attn_v298_q1occ4_st2_np128_90a"',
    'name="okc_attn_v298_q1occ4_st2_np64_90a"',
    1,
)
if np64 == text:
    raise SystemExit("q1 occ4 NP64 anchor changed")
out64 = ROOT / "submission_sweep_q1_occ4_st2_np64.py"
out64.write_text(np64, encoding="utf-8", newline="\n")
print(out64)
