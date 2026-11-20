"""Reconstruct the immutable source lineage of passing check cf1a10603bba46ee.

The archived CUDA build immediately preceding that job records the incumbent
with q_len=1 MINB=3, a three-SM launch target, and the original 128 producers.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
DESTINATION = ROOT / "submission_recovered_cf1a.py"

REPLACEMENTS = (
    (
        "#ifndef OKC_WSMINB\n#define OKC_WSMINB 2\n#endif",
        "#ifndef OKC_WSMINB\n#define OKC_WSMINB 2\n#endif\n"
        "#ifndef OKC_WSMINB1\n"
        "#define OKC_WSMINB1 3   // q_len 1 flat: three resident CTAs, retaining all four producer warps\n"
        "#endif",
    ),
    (
        "  static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow",
        "  static constexpr int MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow",
    ),
    (
        'name="okc_attn_v298_integrated_90a"',
        'name="okc_attn_v298_q1occ3_np128_90a"',
    ),
    (
        "        ft = sms if (L == 8 and OKC_F2) else target",
        "        # q_len=1 flat is launch-bounded for three resident CTAs while retaining 128 producers.\n"
        "        ft = sms if (L == 8 and OKC_F2) else ((3 * sms) if L == 1 else target)",
    ),
)

text = SOURCE.read_text(encoding="utf-8")
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one anchor, found {text.count(old)}: {old[:80]!r}")
    text = text.replace(old, new, 1)

DESTINATION.write_text(text, encoding="utf-8", newline="\n")
print(DESTINATION)
