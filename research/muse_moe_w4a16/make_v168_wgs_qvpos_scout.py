"""Generate v168 scouts that avoid final-pass dynamic indexing of local qv[8]."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v139_wgs_kperm3.py"
DST = ROOT / "submission_v168_wgs_qvpos_scout.py"


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
        "#define OKC_WGSQVPOS 0  // dev: 1=leader route from shared, 2=all final-pass route uses from shared\n"
        "#endif\n",
        "v168 macro",
    )
    text = replace_once(
        text,
        "      int qv[8];\n"
        "#pragma unroll\n"
        "      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n",
        "#if OKC_WGSQVPOS != 2\n"
        "      int qv[8];\n"
        "#pragma unroll\n"
        "      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n"
        "#endif\n",
        "conditional local route array",
    )
    text = replace_once(
        text,
        "        ps[sl] = 0.f;\n"
        "        if (qv[sl] >= 0) {\n"
        "          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n",
        "        ps[sl] = 0.f;\n"
        "#if OKC_WGSQVPOS == 2\n"
        "        const int q = S.pos[tk * 8 + sl];\n"
        "        if (q >= 0) {\n"
        "          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 2;\n"
        "#else\n"
        "        if (qv[sl] >= 0) {\n"
        "          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
        "#endif\n",
        "first route use",
    )
    text = replace_once(
        text,
        "      for (int sl = 0; sl < 8; ++sl)\n"
        "        if (qv[sl] >= 0) {  // uniform over the CTA (one token)\n",
        "      for (int sl = 0; sl < 8; ++sl)\n"
        "#if OKC_WGSQVPOS == 2\n"
        "        if (S.pos[tk * 8 + sl] >= 0) {  // uniform over the CTA (one token)\n"
        "#else\n"
        "        if (qv[sl] >= 0) {  // uniform over the CTA (one token)\n"
        "#endif\n",
        "shuffle route use",
    )
    text = replace_once(
        text,
        "        red[NW * 8 + tid] = qv[tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n",
        "#if OKC_WGSQVPOS\n"
        "        red[NW * 8 + tid] = S.pos[tk * 8 + tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n"
        "#else\n"
        "        red[NW * 8 + tid] = qv[tid] >= 0 ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f) : 0.f;\n"
        "#endif\n",
        "leader route use",
    )
    text = replace_once(
        text,
        "      for (int sl = 0; sl < 8; ++sl)\n"
        "        if (qv[sl] >= 0) {\n"
        "          const float r = red[NW * 8 + sl];\n",
        "      for (int sl = 0; sl < 8; ++sl)\n"
        "#if OKC_WGSQVPOS == 2\n"
        "        if (S.pos[tk * 8 + sl] >= 0) {\n"
        "#else\n"
        "        if (qv[sl] >= 0) {\n"
        "#endif\n"
        "          const float r = red[NW * 8 + sl];\n",
        "reduction route use",
    )

    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
