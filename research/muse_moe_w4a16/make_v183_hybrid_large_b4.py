"""Build v183: exact v178 plus S052 large-B4 for 33 <= M <= 256."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s052_large_b4_partial_norm.py"
DESTINATION = ROOT / "submission_v183_hybrid_large_b4.py"


def literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {path.name}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = BASE.read_text()
    source = literal_assignment(DECODER, "MOE_CUDA_SRC")
    defines = tuple(literal_assignment(DECODER, "MOE_DEFINES"))
    assert "#define OKC_BIGB4 0" in source
    assert "large B4FIN build supports M <= 256" in source
    assert defines == ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")

    marker = "V167_MAIN_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1')\n"
    text = replace_once(
        text,
        marker,
        f"ALT_SRCS['s052_large_b4'] = {source!r}\n"
        f"S052_MAIN_DEFINES = {defines!r}\n"
        + marker,
        "S052 source and defines",
    )

    text = replace_once(
        text,
        '    v167_small = any(d.startswith("@v167_") for d in defines)\n'
        '    if not v167_small:\n'
        '        defines = defines + ("-DOKC_MULDEC=" + str(int(use_mul)),)\n',
        '    v167_small = any(d.startswith("@v167_") for d in defines)\n'
        '    s052_exact = "@s052_large_b4" in defines\n'
        '    if not v167_small and not s052_exact:\n'
        '        defines = defines + ("-DOKC_MULDEC=" + str(int(use_mul)),)\n',
        "exact S052 define policy",
    )
    text = replace_once(
        text,
        '                               *([] if v167_small else ["--use_fast_math", "-Xptxas", "-v,-O3"]),\n',
        '                               *([] if v167_small or s052_exact else ["--use_fast_math", "-Xptxas", "-v,-O3"]),\n',
        "exact S052 compiler policy",
    )

    prepare_anchor = '''def prepare(weights):
    tiny = _prepare_layout(weights, False)
    m32 = _prepare_v167_m32_layout(weights, tiny)
    larger = _prepare_layout(weights, True)
    return {"tiny": tiny, "m32": m32, "larger": larger}
'''
    prepare_replacement = '''def _prepare_s052_large_b4_layout(weights, base):
    """Exact S052 non-SC6 KPERM3 WGS weights and private v139-compatible scratch."""
    dev = weights["w13"].device
    lo, cnt = base["lo"], base["cnt"]
    I = weights["w13"].shape[1] // 2
    H = weights["w2"].shape[1]
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    assert _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state = dict(base)
    state["w"] = list(base["w"][:12]) + [
        torch.cat([_v167_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), False)
                   for e in range(lo, lo + cnt)]),
        torch.cat([_v167_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, False)
                   for e in range(lo, lo + cnt)]),
    ]
    state["scratch"] = _make_scratch(dev)
    state["sfx"] = ("-DOKC_PRMTSC=1",)
    state["use_mul"] = False
    defs = tuple(S052_MAIN_DEFINES) + state["sfx"] + ("-DOKC_B4FIN=1", "-DOKC_BIGB4=1", "@s052_large_b4")
    _ext(defs, use_mul=False)
    torch.cuda.synchronize()
    state["buf_out"] = torch.empty((MMAX, 4096), dtype=torch.bfloat16, device=dev)
    return state


def prepare(weights):
    tiny = _prepare_layout(weights, False)
    m32 = _prepare_v167_m32_layout(weights, tiny)
    larger = _prepare_layout(weights, True)
    large_b4 = _prepare_s052_large_b4_layout(weights, tiny)
    return {"tiny": tiny, "m32": m32, "larger": larger, "large_b4": large_b4}
'''
    text = replace_once(text, prepare_anchor, prepare_replacement, "four-state prepare")
    text = replace_once(
        text,
        '    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]\n',
        '    state = (state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else\n'
        '             state["large_b4"] if _M <= 256 else state["larger"])\n',
        "large-B4 state dispatch",
    )
    text = replace_once(
        text,
        '    elif M <= 32:\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n'
        '    else:\n',
        '    elif M <= 32:\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n'
        '    elif M <= 256:\n'
        '        ext = _ext(tuple(S052_MAIN_DEFINES) + sfx + ("-DOKC_B4FIN=1", "-DOKC_BIGB4=1", "@s052_large_b4"), use_mul=False)\n'
        '    else:\n',
        "large-B4 extension dispatch",
    )

    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
