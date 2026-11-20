"""Generate a v167 scout that packs four long-lived WGS pending fields into u32."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v139_wgs_kperm3.py"
DST = ROOT / "submission_v167_wgs_pendpack_scout.py"


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
        "#ifndef OKC_WGSPENDPACK\n#define OKC_WGSPENDPACK 0  // dev: pack deferred WGS state into one long-lived register\n#endif\n",
        "v167 macro",
    )
    text = replace_once(
        text,
        "struct WgsPend { int kind, u, slot, nst; };\n",
        "struct WgsPend { int kind, u, slot, nst; };\n"
        "__device__ __forceinline__ uint32_t wgs_pend_pack(int kind, int u, int slot, int nst) {\n"
        "  return (uint32_t)(kind & 3) | ((uint32_t)(u & 8191) << 2)\n"
        "       | ((uint32_t)((slot + 1) & 255) << 15) | ((uint32_t)(nst & 31) << 23);\n"
        "}\n",
        "packed pending helper",
    )

    packed_flush = r'''#if OKC_WGSPENDPACK
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, uint32_t& pd) {
  const int kind = pd & 3;
  if (kind == 0) return;
  const int u = (pd >> 2) & 8191;
  const int slot = ((pd >> 15) & 255) - 1;
  const int nst = (pd >> 23) & 31;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" : : "r"(3 + wg) : "memory");
  if (kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" : : "l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
  if constexpr (DF) {
    if (kind == 2) {
      if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" : : "l"(fc), "r"(nst) : "memory");
      pd = 0;
      return;
    }
    if (wt == 0) { while (ld_acquire(fc) < 16 - nst) __nanosleep(32); }
    asm volatile("bar.sync %0, 128;\n" : : "r"(3 + wg) : "memory");
    pd = 0;
  } else {
    if (wt == 0) {
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(nst) : "memory");
      S.wglast[wg] = old + nst == 16;
    }
    asm volatile("bar.sync %0, 128;\n" : : "r"(3 + wg) : "memory");
    const bool last = S.wglast[wg];
    pd = 0;
    if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" : : : "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" : : "r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" : : "l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
#endif
'''
    text = replace_once(
        text,
        "// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)\n",
        packed_flush
        + "// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)\n",
        "packed flush overload",
    )
    text = replace_once(
        text,
        "template <bool PB, int NTT>\n"
        "__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n"
        "                                        int& seq, WgsPend& pd) {\n",
        "template <bool PB, int NTT>\n"
        "__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n"
        "                                        int& seq,\n"
        "#if OKC_WGSPENDPACK\n"
        "                                        uint32_t& pd) {\n"
        "#else\n"
        "                                        WgsPend& pd) {\n"
        "#endif\n",
        "conditional pending type",
    )
    text = replace_once(
        text,
        "        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n",
        "#if OKC_WGSPENDPACK\n"
        "        if (FIRST && ci == OKC_WGSFCI && (pd & 3)) wgs_flush<PB>(p, S, pd);\n"
        "#else\n"
        "        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n"
        "#endif\n",
        "packed deferred flush test",
    )
    text = replace_once(
        text,
        "    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};\n",
        "#if OKC_WGSPENDPACK\n"
        "    pd = wgs_pend_pack((((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0);\n"
        "#else\n"
        "    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};\n"
        "#endif\n",
        "packed split assignment",
    )
    text = replace_once(
        text,
        "  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred\n",
        "#if OKC_WGSPENDPACK\n"
        "  if (!PB) pd = wgs_pend_pack(1, u, -1, 0);  // the doneA release waits until the stores are done: deferred\n"
        "#else\n"
        "  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred\n"
        "#endif\n",
        "packed whole-unit assignment",
    )
    text = replace_once(
        text,
        "  WgsPend pd{0, 0, 0, 0};\n",
        "#if OKC_WGSPENDPACK\n"
        "  uint32_t pd = 0;\n"
        "#else\n"
        "  WgsPend pd{0, 0, 0, 0};\n"
        "#endif\n",
        "packed declaration",
    )

    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
