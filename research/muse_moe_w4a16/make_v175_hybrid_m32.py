"""Build v175: v167 small/M32 paths plus frozen-v139 M>=33 WGS.

The M32 state uses v167's exact SC6/KPERM WGS packing and exact main CUDA
source.  M<=16 keeps the validated v172 small paths; M>=33 keeps v139's
exact CUDA source, defines, _POS_MUL packing, and dispatch.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v172_hybrid_small.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v175_hybrid_m32.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def literal_assignments(path: Path) -> dict[str, object]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def function_source(path: Path, name: str, rename: str | None = None) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"{path.name}:{name}: expected one function, found {len(matches)}")
    source = ast.get_source_segment(text, matches[0])
    assert source is not None
    if rename is not None:
        source = source.replace(f"def {name}(", f"def {rename}(", 1)
    return source


def main() -> None:
    text = BASE.read_text()
    advanced = literal_assignments(ADVANCED)
    main_source = advanced["CUDA_SRC"]
    main_defines = tuple(advanced["MAIN_DEFINES"])

    text = replace_once(
        text,
        "\n\n# Bit layout of a 32-bit word holding 8 E2M1 codes",
        f"\nALT_SRCS['v167_main'] = {main_source!r}\n"
        f"V167_MAIN_DEFINES = {main_defines!r}\n\n"
        "# Bit layout of a 32-bit word holding 8 E2M1 codes",
        "v167 main source and defines",
    )

    packing = "\n\n".join((
        function_source(ADVANCED, "_sc6_ok"),
        function_source(ADVANCED, "_sc6_wgs"),
        function_source(ADVANCED, "_kperm_src"),
        function_source(ADVANCED, "_kperm_codes"),
        function_source(ADVANCED, "_kperm_rows"),
        function_source(ADVANCED, "_wgs_copy", "_v167_wgs_copy"),
    ))
    text = replace_once(
        text,
        "\n\n# the M >= 64 build when the scale bytes rule out the PRMT decode",
        "\n\n" + packing +
        "\n\n# the M >= 64 build when the scale bytes rule out the PRMT decode",
        "v167 SC6/KPERM packing",
    )

    old_prepare = (
        'def prepare(weights):\n'
        '    return {"tiny": _prepare_layout(weights, False), "larger": _prepare_layout(weights, True)}\n'
    )
    new_prepare = '''def _prepare_v167_m32_layout(weights, base):
    """Build only v167's SC6 WGS copies; share the exact _POS base tensors."""
    dev = weights["w13"].device
    lo, cnt = base["lo"], base["cnt"]
    I = weights["w13"].shape[1] // 2
    H = weights["w2"].shape[1]
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    assert _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    assert _sc6_ok(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state = dict(base)
    state["w"] = list(base["w"][:12]) + [
        torch.cat([_v167_wgs_copy(weights["w13"][e], weights["w13_sf"][e], _kperm_rows(rowsA), True)
                   for e in range(lo, lo + cnt)]),
        torch.cat([_v167_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, True)
                   for e in range(lo, lo + cnt)]),
    ]
    state["scratch"] = _make_scratch(dev)
    state["sfx"] = ("-DOKC_PRMTSC=1", "-DOKC_SC6W=1")
    state["use_mul"] = False
    defs = tuple(V167_MAIN_DEFINES) + state["sfx"] + ("@v167_main",)
    _ext(defs, use_mul=False)
    torch.cuda.synchronize()
    state["buf_out"] = torch.empty((MMAX, 4096), dtype=torch.bfloat16, device=dev)
    return state


def prepare(weights):
    tiny = _prepare_layout(weights, False)
    m32 = _prepare_v167_m32_layout(weights, tiny)
    larger = _prepare_layout(weights, True)
    return {"tiny": tiny, "m32": m32, "larger": larger}
'''
    text = replace_once(text, old_prepare, new_prepare, "three-state prepare")
    text = replace_once(
        text,
        '    state = state["tiny"] if data["hidden_states"].shape[0] <= MID_MAX_M else state["larger"]\n',
        '    _M = data["hidden_states"].shape[0]\n'
        '    state = state["tiny"] if _M <= MID_MAX_M else state["m32"] if _M <= 32 else state["larger"]\n',
        "three-state dispatch",
    )
    text = replace_once(
        text,
        '    elif MID_DEFINES and M <= MID_MAX_M:\n'
        '        ext = _ext(tuple(MID_DEFINES) + sfx, use_mul=state["use_mul"])\n'
        '    else:\n'
        '        ext = _ext(tuple(_main_defines(sfx)) + sfx, use_mul=state["use_mul"])\n',
        '    elif MID_DEFINES and M <= MID_MAX_M:\n'
        '        ext = _ext(tuple(MID_DEFINES) + sfx, use_mul=state["use_mul"])\n'
        '    elif M <= 32:\n'
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n'
        '    else:\n'
        '        ext = _ext(tuple(_main_defines(sfx)) + sfx, use_mul=state["use_mul"])\n',
        "M32 extension dispatch",
    )

    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
