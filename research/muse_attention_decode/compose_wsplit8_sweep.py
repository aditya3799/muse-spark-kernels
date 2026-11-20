"""Generate exact q8-only static paired-CTA work-split variants."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
base = SOURCE.read_text(encoding="utf-8")

macro_anchor = "#define OKC_WSPLIT 500  // flat: per mille of the keys for the first half of the grid (500 = even)\n#endif"
code_anchor = """  if (OKC_WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
    // CTAs c and c + G/2 are dispatched to the same SM, and the first one dispatched runs faster: give the first
    // half of the grid WSPLIT per mille of the keys
    const int64_t Th = T * OKC_WSPLIT / 1000, H = G >> 1;"""

if base.count(macro_anchor) != 1 or base.count(code_anchor) != 1:
    raise SystemExit("WSPLIT anchors changed")

for split in (535, 555, 575):
    text = base.replace(
        macro_anchor,
        macro_anchor + f"\n#ifndef OKC_WSPLIT8\n#define OKC_WSPLIT8 {split}  // q8-only static paired-CTA share; q1 retains its FPART path\n#endif",
        1,
    )
    text = text.replace(
        code_anchor,
        """  constexpr int WSPLIT = (L == 8) ? OKC_WSPLIT8 : OKC_WSPLIT;
  if (WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
    // CTAs c and c + G/2 are dispatched to the same SM, and the first one dispatched runs faster: give the first
    // half of the grid WSPLIT per mille of the keys
    const int64_t Th = T * WSPLIT / 1000, H = G >> 1;""",
        1,
    )
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_wsplit8_{split}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_wsplit8_{split}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
