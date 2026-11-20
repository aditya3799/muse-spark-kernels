"""Generate exact pipeline/scheduling micro-variants from the active incumbent."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
base = (ROOT / "submission.py").read_text(encoding="utf-8")

variants = {
    "unr8_4": (
        "#define OKC_UNR8 2      // q_len 8 flat loop unroll",
        "#define OKC_UNR8 4      // q_len 8 flat loop unroll",
    ),
    "pvsplit_2": (
        "#define OKC_PVSPLIT 1   // q8 PV:",
        "#define OKC_PVSPLIT 2   // q8 PV:",
    ),
    "ptpf_0": (
        "#define OKC_PTPF 1      // (with OKC_FPART)",
        "#define OKC_PTPF 0      // (with OKC_FPART)",
    ),
}

for tag, (old, new) in variants.items():
    if base.count(old) != 1:
        raise SystemExit(f"anchor changed for {tag}: {base.count(old)}")
    text = base.replace(old, new, 1)
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_micro_{tag}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_micro_{tag}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
