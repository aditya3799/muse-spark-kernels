"""Generate the v165 WGS epilogue metadata/address scout from frozen v139."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v139_wgs_kperm3.py"
DST = ROOT / "submission_v165_wgs_epimeta_scout.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_block(text: str, start: str, end: str, new: str, label: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"{label}: non-unique boundary")
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + new + text[j:]


META_HELPER = r'''#if OKC_WGSEPIMETA
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile_meta(const Params& p, int cnt, int jst, int e, int jrow, int ntt,
                                                   int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (!PB) {
    const int col = r16 * 8 + g;
#if OKC_KPERM3
    const int pcol = wgs_kperm3_col(col);
#else
    const int pcol = col;
#endif
    const float gsc = p.gsA[e] * TWO62;
#if OKC_WGSEPIADDR
    const int tok0 = nt * 8 + 2 * t;
    const size_t oi = ((((size_t)(pcol >> 8) * ntt + nt) * 32 + ((pcol & 255) >> 3)) * 64
                       + (size_t)(2 * t) * 8 + (pcol & 7));
    __nv_bfloat16* dst = p.xB + (size_t)jrow * HD + oi;
    if (tok0 < cnt) {
      const float gt = d4[0] * gsc, up = d4[2] * gsc;
      const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(dst, __float2bfloat16(probe_q(v)));
    }
    if (tok0 + 1 < cnt) {
      const float gt = d4[1] * gsc, up = d4[3] * gsc;
      const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(dst + 8, __float2bfloat16(probe_q(v)));
    }
#else
    __nv_bfloat16* xb = p.xB + (size_t)jrow * HD + (pcol & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((pcol >> 8) * ntt + (tk >> 3)) * 32 + ((pcol & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
#endif
  } else {
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc;
    const float w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#if OKC_WGSEPIADDR
    float* y = p.ybuf + (size_t)(jst + tok0) * HD + n0;
    if (tok0 < cnt) {
      st_keep_f32(y, d4[0] * w0);
      st_keep_f32(y + 8, d4[2] * w1);
    }
    if (tok0 + 1 < cnt) {
      y += HD;
      st_keep_f32(y, d4[1] * w0);
      st_keep_f32(y + 8, d4[3] * w1);
    }
#else
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
#endif
  }
}
#endif
'''


def main() -> None:
    text = SRC.read_text()
    text = replace_once(
        text,
        "#ifndef OKC_WGSDBG\n#define OKC_WGSDBG 0  // dev: WGS waits time out with a printf + trap (tag says which wait)\n#endif\n",
        "#ifndef OKC_WGSDBG\n#define OKC_WGSDBG 0  // dev: WGS waits time out with a printf + trap (tag says which wait)\n#endif\n"
        "#ifndef OKC_WGSEPIMETA\n#define OKC_WGSEPIMETA 0  // dev: load WGS job metadata once per completed segment\n#endif\n"
        "#ifndef OKC_WGSEPIADDR\n#define OKC_WGSEPIADDR 0  // dev: strength-reduce WGS epilogue row addresses (requires EPIMETA)\n#endif\n"
        "#if OKC_WGSEPIADDR && !OKC_WGSEPIMETA\n#error \"OKC_WGSEPIADDR requires OKC_WGSEPIMETA\"\n#endif\n",
        "v165 macros",
    )
    start = "template <bool PB>\n__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {\n"
    end = "template <bool PB>\n__device__ __forceinline__ void wgs_flush"
    old = text[text.index(start):text.index(end, text.index(start))]
    text = replace_block(text, start, end, META_HELPER + old, "epilogue helper")
    text = replace_once(
        text,
        "  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n"
        "#pragma unroll\n"
        "  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n",
        "  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n"
        "#if OKC_WGSEPIMETA\n"
        "  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j], jrow = S.jrow[j];\n"
        "#pragma unroll\n"
        "  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile_meta<PB>(p, cnt, jst, e, jrow, NTT, r16, nt, d[nt]);\n"
        "#else\n"
        "#pragma unroll\n"
        "  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n"
        "#endif\n",
        "whole-unit epilogue calls",
    )
    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
