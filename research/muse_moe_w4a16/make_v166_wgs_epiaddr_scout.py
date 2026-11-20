"""Generate the v166 address-only WGS epilogue scout from frozen v139."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v139_wgs_kperm3.py"
DST = ROOT / "submission_v166_wgs_epiaddr_scout.py"


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
        "#ifndef OKC_WGSEPIADDR\n#define OKC_WGSEPIADDR 0  // dev: strength-reduce WGS epilogue row addresses\n#endif\n",
        "v166 macro",
    )
    text = replace_once(
        text,
        "    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (pcol & 7);\n"
        "#pragma unroll\n"
        "    for (int h = 0; h < 2; ++h) {\n"
        "      const int tk = nt * 8 + 2 * t + h;\n"
        "      if (tk < cnt) {\n"
        "        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\n"
        "        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n"
        "        st_keep_bf16(xb + (size_t)((((pcol >> 8) * ntt + (tk >> 3)) * 32 + ((pcol & 255) >> 3)) * 64 + (tk & 7) * 8),\n"
        "                     __float2bfloat16(probe_q(v)));\n"
        "      }\n"
        "    }\n",
        "    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (pcol & 7);\n"
        "#if OKC_WGSEPIADDR\n"
        "    const int tok0 = nt * 8 + 2 * t;\n"
        "    const size_t oi = ((((size_t)(pcol >> 8) * ntt + nt) * 32 + ((pcol & 255) >> 3)) * 64\n"
        "                       + (size_t)(2 * t) * 8);\n"
        "    __nv_bfloat16* dst = xb + oi;\n"
        "    if (tok0 < cnt) {\n"
        "      const float gt = d4[0] * gsc, up = d4[2] * gsc;\n"
        "      const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n"
        "      st_keep_bf16(dst, __float2bfloat16(probe_q(v)));\n"
        "    }\n"
        "    if (tok0 + 1 < cnt) {\n"
        "      const float gt = d4[1] * gsc, up = d4[3] * gsc;\n"
        "      const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n"
        "      st_keep_bf16(dst + 8, __float2bfloat16(probe_q(v)));\n"
        "    }\n"
        "#else\n"
        "#pragma unroll\n"
        "    for (int h = 0; h < 2; ++h) {\n"
        "      const int tk = nt * 8 + 2 * t + h;\n"
        "      if (tk < cnt) {\n"
        "        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\n"
        "        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n"
        "        st_keep_bf16(xb + (size_t)((((pcol >> 8) * ntt + (tk >> 3)) * 32 + ((pcol & 255) >> 3)) * 64 + (tk & 7) * 8),\n"
        "                     __float2bfloat16(probe_q(v)));\n"
        "      }\n"
        "    }\n"
        "#endif\n",
        "phase-A address reduction",
    )
    text = replace_once(
        text,
        "#pragma unroll\n"
        "    for (int h = 0; h < 2; ++h) {\n"
        "      const int tk = tok0 + h;\n"
        "      if (tk < cnt) {\n"
        "        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n"
        "        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n"
        "      }\n"
        "    }\n",
        "#if OKC_WGSEPIADDR\n"
        "    float* y = p.ybuf + (size_t)(jst + tok0) * HD + n0;\n"
        "    if (tok0 < cnt) {\n"
        "      st_keep_f32(y, d4[0] * w0);\n"
        "      st_keep_f32(y + 8, d4[2] * w1);\n"
        "    }\n"
        "    if (tok0 + 1 < cnt) {\n"
        "      y += HD;\n"
        "      st_keep_f32(y, d4[1] * w0);\n"
        "      st_keep_f32(y + 8, d4[3] * w1);\n"
        "    }\n"
        "#else\n"
        "#pragma unroll\n"
        "    for (int h = 0; h < 2; ++h) {\n"
        "      const int tk = tok0 + h;\n"
        "      if (tk < cnt) {\n"
        "        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n"
        "        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n"
        "      }\n"
        "    }\n"
        "#endif\n",
        "phase-B address reduction",
    )
    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
