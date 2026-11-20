"""Build v181: v178 plus the decoder winner's B4FIN path at M=32 only."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v178_hybrid_m32_retry.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
DECODER = ROOT.parent / "muse_decode_layer" / "submission_s082_keell0renz_latest.py"
DESTINATION = ROOT / "submission_v181_hybrid_m32_b4fin.py"


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
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


def section(text: str, start: str, end: str, start_at: int = 0) -> str:
    i = text.index(start, start_at)
    j = text.index(end, i)
    return text[i:j]


def b4fin_source() -> str:
    source = literal_assignment(ADVANCED, "CUDA_SRC")
    decoder = literal_assignment(DECODER, "MOE_CUDA_SRC")

    macro = section(
        decoder,
        "#ifndef OKC_B4FIN\n",
        "#ifndef OKC_FINCOL\n",
    )
    source = replace_once(
        source,
        "#ifndef OKC_FINCOL\n",
        macro + "#ifndef OKC_FINCOL\n",
        "B4FIN compile-time contract",
    )

    source = replace_once(
        source,
        "  unsigned long long* dbg;\n};",
        "  unsigned long long* dbg;\n"
        "  float* sspart;  // B4FIN: per-(route, 16-column tile) sum-of-squares partials\n"
        "};",
        "B4FIN parameter",
    )

    epi_start = decoder.index("#if OKC_B4FIN\n    {  // bet4_claude: this tile's 16-column sum of squares")
    epi = section(decoder, "#if OKC_B4FIN\n", "  }\n}\ntemplate <bool PB>", epi_start)
    epi_anchor = "    }\n  }\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush"
    source = replace_once(
        source,
        epi_anchor,
        "    }\n" + epi + "  }\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush",
        "phase-B fixed norm partials",
    )

    final_comment = decoder.index("// final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]")
    final = section(
        decoder,
        "#if OKC_WGS && OKC_B4FIN\n",
        "#elif OKC_WGS && OKC_WGSFINSS\n",
        final_comment,
    )
    source_comment = source.index("// final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]")
    old_final = "#if OKC_WGS && OKC_WGSFINSS\n"
    assert source.count(old_final, source_comment) == 1
    source = source[:source_comment] + source[source_comment:].replace(
        old_final, final + "#elif OKC_WGS && OKC_WGSFINSS\n", 1
    )

    host_start = decoder.index("#if OKC_WGS && OKC_B4FIN\n  // B4FIN runs only at M <= 32")
    host = section(decoder, "#if OKC_WGS && OKC_B4FIN\n", "  p.inter =", host_start)
    source = replace_once(
        source,
        "  p.ybuf = p.ss + okc::EMAX;\n  p.inter =",
        "  p.ybuf = p.ss + okc::EMAX;\n" + host + "  p.inter =",
        "B4FIN scratch alias",
    )

    assert source.count("#ifndef OKC_B4FIN") == 1
    assert source.count("p.sspart") == 5
    assert source.count("#if OKC_WGS && OKC_B4FIN") == 2
    assert "static_assert(NT == 512 && NW == 16" in source
    return source


def main() -> None:
    text = BASE.read_text()
    b4 = b4fin_source()
    marker = "V167_MAIN_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1')\n"
    text = replace_once(
        text,
        marker,
        f"ALT_SRCS['v181_b4fin'] = {b4!r}\n" + marker,
        "B4FIN alternate source",
    )
    text = replace_once(
        text,
        '    defs = tuple(V167_MAIN_DEFINES) + state["sfx"] + ("@v167_main",)\n',
        '    defs = tuple(V167_MAIN_DEFINES) + state["sfx"] + ("-DOKC_B4FIN=1", "@v181_b4fin")\n',
        "M32 prepare build",
    )
    text = replace_once(
        text,
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("@v167_main",), use_mul=False)\n',
        '        ext = _ext(tuple(V167_MAIN_DEFINES) + sfx + ("-DOKC_B4FIN=1", "@v181_b4fin"), use_mul=False)\n',
        "M32 timed dispatch",
    )
    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
