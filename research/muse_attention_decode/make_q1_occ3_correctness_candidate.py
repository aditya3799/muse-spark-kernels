"""Build the correctness-first q1 occupancy candidate.

The q_len=8 F2 family is deliberately excluded.  Its guarded two-stream
variants still failed four full-attention correctness cases on H200, while
the original q_len=8 topology is the established correct path.  Keep the
q_len=1 occupancy work and give this rollback candidate a distinct extension
name so it cannot reuse an experimental F2 build.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_q1_occ3_np128.py"
DESTINATION = ROOT / "submission_q1_occ3_q8_correct.py"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    required = {
        "#define OKC_F2 0": "CUDA F2 must remain disabled",
        "OKC_FLATCOOP = 1": "the proven flat q8 finalizer must remain enabled",
        "OKC_F2 = 0": "the Python launch plan must match CUDA",
    }
    for anchor, message in required.items():
        if anchor not in text:
            raise RuntimeError(f"{message}: missing {anchor!r}")

    old_name = 'name="okc_attn_v298_q1occ3_np128_90a"'
    new_name = 'name="okc_attn_v298_q1occ3_q8correct_90a"'
    if text.count(old_name) != 1:
        raise RuntimeError("unexpected extension-name anchor count")
    text = text.replace(old_name, new_name, 1)

    with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(DESTINATION)


if __name__ == "__main__":
    main()
