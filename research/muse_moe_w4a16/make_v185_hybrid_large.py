"""Build the narrow production hybrid proven by the v184 Muse factorial."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v178_hybrid_m32_retry.py")
OUT = Path(__file__).with_name("submission_v185_hybrid_m32_m128_m256.py")


def main():
    source = BASE.read_text()
    old_state = (
        '    state = state["tiny"] if _M <= MID_MAX_M else '
        'state["m32"] if _M <= 32 else state["larger"]'
    )
    new_state = (
        '    state = state["tiny"] if _M <= MID_MAX_M else '
        'state["m32"] if (_M <= 32 or _M in (128, 256)) else state["larger"]'
    )
    old_ext = "    elif M <= 32:\n"
    new_ext = "    elif M <= 32 or M in (128, 256):\n"
    assert source.count(old_state) == 1
    assert source.count(old_ext) == 1
    candidate = source.replace(old_state, new_state).replace(old_ext, new_ext)
    assert candidate.count(new_state) == 1
    assert candidate.count(new_ext) == 1
    OUT.write_text(candidate)
    print(OUT)


if __name__ == "__main__":
    main()
