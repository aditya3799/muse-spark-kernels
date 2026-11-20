"""Generate an exact candidate that disables q1 page-table prefetch only for B32."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
text = (ROOT / "submission.py").read_text(encoding="utf-8")
old = "if (OKC_PTPF && mi != 0u && csf < cef)"
new = "if (OKC_PTPF && p.B != 32 && mi != 0u && csf < cef)"

if text.count(old) != 1:
    raise SystemExit(f"PTPF condition anchor changed: {text.count(old)}")

text = text.replace(old, new, 1).replace(
    'name="okc_attn_v298_q1occ3_np128_90a"',
    'name="okc_attn_v298_q1occ3_ptpf_b32off_90a"',
    1,
)
out = ROOT / "submission_sweep_ptpf_b32off.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)
