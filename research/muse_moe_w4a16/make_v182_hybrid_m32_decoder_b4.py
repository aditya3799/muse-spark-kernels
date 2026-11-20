"""Build v182: exact decoder-winner B4FIN source only for v178's M32 state."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s082_keell0renz_latest.py"
DESTINATION = ROOT / "submission_v182_hybrid_m32_decoder_b4.py"


def assignment(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {path.name}")


def function_source(path: Path, name: str) -> str:
    text = path.read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"{path.name}:{name}: expected one function, found {len(nodes)}")
    source = ast.get_source_segment(text, nodes[0])
    assert source is not None
    return source


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = BASE.read_text()
    source = assignment(DECODER, "MOE_CUDA_SRC")
    assert "constexpr int HD = 4096, ID = 4096" in source
    assert "#if OKC_WGS && OKC_B4FIN" in source
    assert "wgs_ranges(p, S);  // all 32 lanes" in source
    for name in ("_sc6_ok", "_sc6_wgs", "_kperm_src", "_kperm_codes", "_kperm_rows", "_wgs_copy"):
        assert function_source(ADVANCED, name) == function_source(DECODER, name), name

    marker = "V167_MAIN_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1')\n"
    text = replace_once(
        text,
        marker,
        f"ALT_SRCS['v182_decoder_b4'] = {source!r}\n" + marker,
        "exact decoder B4 source",
    )
    text = replace_once(
        text,
        '    defs = tuple(V167_MAIN_DEFINES) + state["sfx"] + ("@v167_main",)\n',
        '    defs = tuple(V167_MAIN_DEFINES) + state["sfx"] + ("-DOKC_B4FIN=1", "@v182_decoder_b4")\n',
        "M32 prepare build",
    )
    text = replace_once(
        text,
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n',
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("-DOKC_B4FIN=1", "@v182_decoder_b4"), use_mul=False)\n',
        "M32 timed dispatch",
    )
    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
