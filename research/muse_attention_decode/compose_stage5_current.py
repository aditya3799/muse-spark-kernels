"""Graft the audited B32-only five-stage q8 kernel onto today's incumbent settings."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
text = (ROOT / "submission_q8_st5_vt1.py").read_text(encoding="utf-8")

replacements = {
    "#define OKC_WSMINB 2\n#endif":
        "#define OKC_WSMINB 2\n#endif\n#ifndef OKC_WSMINB1\n"
        "#define OKC_WSMINB1 3   // q_len 1 flat: three resident CTAs\n#endif",
    "static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);":
        "static constexpr int MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);",
    'asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" :―"r"(dst), "l"(src), "r"(src_bytes));':
        'asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16, %2;\\n" :―"r"(dst), "l"(src), "r"(src_bytes));',
    "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target":
        "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)",
    '_build(tag="_q8_exactst4_deepb32", flags=("-DOKC_CONT=1",))':
        '_build(tag="_q1occ3_l264_q8deepb32", flags=("-DOKC_CONT=1",))',
}

# Keep the CUDA operand separator ASCII; the visually distinct marker above avoids
# accidental source rewriting by this generator's own Python parser.
replacements = {old.replace(":―", "::"): new.replace(":―", "::") for old, new in replacements.items()}

for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"stage5/current anchor changed: {old!r} -> {text.count(old)}")
    text = text.replace(old, new, 1)

out = ROOT / "submission_sweep_stage5_current.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)
