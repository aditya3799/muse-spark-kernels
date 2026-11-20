"""Build the isolated q_len=1 three-CTA occupancy candidate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
DESTINATION = ROOT / "submission_q1_occ3.py"


REPLACEMENTS = (
    (
        "#define OKC_NP 128      // producer threads (q_len 1 flat; 128 since v111: f16 -0.7%, f32 -0.4% vs 256)",
        "#define OKC_NP 64       // q_len 1 flat: two producer warps; three resident CTAs provide six producer warps / SM",
    ),
    (
        "#ifndef OKC_WSMINB\n#define OKC_WSMINB 2\n#endif",
        "#ifndef OKC_WSMINB\n#define OKC_WSMINB 2\n#endif\n#ifndef OKC_WSMINB1\n#define OKC_WSMINB1 3   // q_len 1 flat: 3 x (128 consumers + 64 producers) fits registers and shared memory on H200\n#endif",
    ),
    (
        "  static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow",
        "  static constexpr int MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow",
    ),
    (
        "        ft = sms if (L == 8 and OKC_F2) else target",
        "        # The q_len=1 flat specialization uses 128 consumers + 64 producers and is launch-bounded for three\n"
        "        # resident CTAs per SM. Give all three slots work; q_len=8 retains its proven two-CTA topology.\n"
        "        ft = sms if (L == 8 and OKC_F2) else ((3 * sms) if L == 1 else target)",
    ),
)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise RuntimeError(f"expected one anchor, found {text.count(old)}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    DESTINATION.write_text(text, encoding="utf-8", newline="\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
