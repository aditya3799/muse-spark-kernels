"""Combine the independently paired M32 B4 and M128/M256 MFIN+SC6 wins."""

from pathlib import Path


BASE = Path(__file__).with_name("submission_v182_hybrid_m32_decoder_b4.py")
OUT = Path(__file__).with_name("submission_v187_combined_b4_large.py")


def main():
    source = BASE.read_text()
    old_compile = "    _ext(defs, use_mul=False)\n    torch.cuda.synchronize()"
    new_compile = (
        "    _ext(defs, use_mul=False)\n"
        "    _ext(tuple(V167_MAIN_DEFINES) + state[\"sfx\"] + (\"@v167_main\",), use_mul=False)\n"
        "    torch.cuda.synchronize()"
    )
    old_state = (
        '    state = state["tiny"] if _M <= MID_MAX_M else '
        'state["m32"] if _M <= 32 else state["larger"]'
    )
    new_state = (
        '    state = state["tiny"] if _M <= MID_MAX_M else '
        'state["m32"] if (_M <= 32 or _M in (128, 256)) else state["larger"]'
    )
    old_dispatch = (
        '    elif M <= 32:\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + '
        '("-DOKC_B4FIN=1", "@v182_decoder_b4"), use_mul=False)\n'
        '    else:\n'
    )
    new_dispatch = (
        '    elif M <= 32:\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + '
        '("-DOKC_B4FIN=1", "@v182_decoder_b4"), use_mul=False)\n'
        '    elif M in (128, 256):\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n'
        '    else:\n'
    )
    for old in (old_compile, old_state, old_dispatch):
        assert source.count(old) == 1, old
    candidate = source.replace(old_compile, new_compile).replace(old_state, new_state).replace(old_dispatch, new_dispatch)
    OUT.write_text(candidate)
    print(OUT)


if __name__ == "__main__":
    main()
