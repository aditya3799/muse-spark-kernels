"""Generate exact q8 flat producer-width variants."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
base = (ROOT / "submission.py").read_text(encoding="utf-8")
anchor = "#define OKC_NP8 128     // q_len 8: 4 producer warps (its consumers need the registers)"

if base.count(anchor) != 1:
    raise SystemExit("NP8 anchor changed")

for np8 in (64, 256):
    text = base.replace(anchor, anchor.replace("128", str(np8), 1), 1)
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_np8_{np8}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_np8_{np8}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
