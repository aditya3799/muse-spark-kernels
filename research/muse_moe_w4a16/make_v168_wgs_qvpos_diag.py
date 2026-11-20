"""Generate the v168 leader-route final-tail A/B diagnostic from v164."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v164_wgs_polysilu_diag.py"
DST = ROOT / "submission_v168_wgs_qvpos_diag.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = SRC.read_text()
    text = replace_once(
        text,
        "#ifndef OKC_WGSDBG\n#define OKC_WGSDBG 0  // dev: WGS waits time out with a printf + trap (tag says which wait)\n#endif\n",
        "#ifndef OKC_WGSDBG\n#define OKC_WGSDBG 0  // dev: WGS waits time out with a printf + trap (tag says which wait)\n#endif\n"
        "#ifndef OKC_WGSQVPOS\n"
        "#define OKC_WGSQVPOS 0  // dev: final normalization leader reads its route directly from shared state\n"
        "#endif\n",
        "v168 macro",
    )
    text = replace_once(
        text,
        "        red[NW * 8 + tid] = qv[tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n",
        "#if OKC_WGSQVPOS\n"
        "        red[NW * 8 + tid] = S.pos[tk * 8 + tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n"
        "#else\n"
        "        red[NW * 8 + tid] = qv[tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n"
        "#endif\n",
        "leader route lookup",
    )
    text = replace_once(
        text,
        'DIAG_VARIANTS = {\n    "poly-silu": tuple(MAIN_DEFINES) + ("-DOKC_WGSPOLY=1",),\n}\n',
        'DIAG_VARIANTS = {\n    "qvpos": tuple(MAIN_DEFINES) + ("-DOKC_WGSQVPOS=1",),\n}\n',
        "diagnostic variant",
    )
    text = replace_once(
        text,
        '                if rel > 0.020 or max_row > 0.060:\n'
        '                    raise RuntimeError(f"{name} output drift: rel={rel:.6f}, max_row={max_row:.6f}")\n',
        '                if not torch.equal(outs[name], outs["main"]):\n'
        '                    mismatches = int(torch.count_nonzero(outs[name] != outs["main"]).item())\n'
        '                    raise RuntimeError(\n'
        '                        f"{name} is not bit-exact: mismatches={mismatches}, "\n'
        '                        f"rel={rel:.6f}, max_row={max_row:.6f}"\n'
        '                    )\n',
        "bit-exact diagnostic gate",
    )
    text = replace_once(
        text,
        '                   max_rel <= 0.020 and max_row <= 0.060)\n',
        '                   max_rel == 0.0 and max_row == 0.0)\n',
        "bit-exact promotion gate",
    )
    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
