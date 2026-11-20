"""Generate pair-plane variants that distribute escape correction over workers.

Each existing compute warp owns a disjoint 32-entry stripe of the escape
table.  Corrections are reduced into a separate shared array, so there is no
dedicated helper warp and no single delayed worker.  The normal packed weight
loads, FP32 FMA order, and compute reduction remain unchanged.  Official
one-hot/FP64 validation is intentionally authoritative because the correction
terms are grouped by worker warp instead of by one helper warp.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_inline_escape_v262.py"
PAIR_MARKER = "# v106: exact C12 row-paired code planes"


def replace_pair_once(text: str, old: str, new: str) -> str:
    head, sep, tail = text.partition(PAIR_MARKER)
    if not sep or tail.count(old) != 1:
        raise RuntimeError(
            f"expected one pair-block match, found {tail.count(old)}: "
            f"{old[:100]!r}")
    return head + sep + tail.replace(old, new, 1)


def distribute_escape(text: str) -> str:
    text = replace_pair_once(
        text,
        "  __shared__ float red[ROWS][NWARP + 1];\n"
        "  if (warp == 0) {\n"
        "    const int2* tb = etab + (size_t)blockIdx.x * pmax;\n"
        "    int2 mine = make_int2(0, 0);\n"
        "    if (lane < pmax) mine = __ldg(&tb[lane]);\n"
        "    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);\n"
        "    float ec[ROWS] = {0.f, 0.f};\n"
        "    for (int i0 = 0; i0 <= cnt; i0 += 32) {\n"
        "      const int i = i0 + lane;\n"
        "      int2 en = mine;\n"
        "      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);\n"
        "      if (i >= 1 && i <= cnt) {\n"
        "        const float c = __bfloat162float(x[en.x & 0xFFFF]) *\n"
        "                        esc_delta((uint32_t)en.y, base4 & 0xFFu);\n"
        "        const int rl = en.x >> 16;\n"
        "        ec[0] += (rl == 0) ? c : 0.f;\n"
        "        ec[1] += (rl == 1) ? c : 0.f;\n"
        "      }\n"
        "    }\n"
        "#pragma unroll\n"
        "    for (int r = 0; r < ROWS; ++r) {\n"
        "      float v = ec[r];\n"
        "#pragma unroll\n"
        "      for (int o = 16; o > 0; o >>= 1)\n"
        "        v += __shfl_xor_sync(0xffffffffu, v, o);\n"
        "      if (lane == 0) red[r][NWARP] = v;\n"
        "    }\n"
        "    __syncwarp();\n"
        "  }\n\n",
        "  __shared__ float red[ROWS][NWARP];\n"
        "  __shared__ float ered[ROWS][NWARP];\n"
        "  const int2* tb = etab + (size_t)blockIdx.x * pmax;\n"
        "  int cnt = (lane == 0) ? __ldg(&tb[0]).x : 0;\n"
        "  cnt = __shfl_sync(0xffffffffu, cnt, 0);\n"
        "  float ec[ROWS] = {0.f, 0.f};\n"
        "  for (int i = 1 + 32 * warp + lane; i <= cnt; i += NTH) {\n"
        "    const int2 en = __ldg(&tb[i]);\n"
        "    const float c = __bfloat162float(x[en.x & 0xFFFF]) *\n"
        "                    esc_delta((uint32_t)en.y, base4 & 0xFFu);\n"
        "    const int rl = en.x >> 16;\n"
        "    ec[0] += (rl == 0) ? c : 0.f;\n"
        "    ec[1] += (rl == 1) ? c : 0.f;\n"
        "  }\n"
        "#pragma unroll\n"
        "  for (int r = 0; r < ROWS; ++r) {\n"
        "    float v = ec[r];\n"
        "#pragma unroll\n"
        "    for (int o = 16; o > 0; o >>= 1)\n"
        "      v += __shfl_xor_sync(0xffffffffu, v, o);\n"
        "    if (lane == 0) ered[r][warp] = v;\n"
        "  }\n\n")
    text = replace_pair_once(
        text,
        "    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];",
        "    for (int q = 0; q < NWARP; ++q) v += red[tid][q];\n"
        "#pragma unroll\n"
        "    for (int q = 0; q < NWARP; ++q) v += ered[tid][q];")
    return text


def inject_helpers(text: str, helpers: str) -> str:
    anchor = "_V106_CG_CUDA = r'''\n"
    if text.count(anchor) != 1:
        raise RuntimeError("pair CUDA marker mismatch")
    return text.replace(anchor, anchor + helpers.strip() + "\n", 1)


def replace_weight_loads(text: str, helper: str) -> str:
    for old, new in (
            ("lw[0][j] = __ldg(pv + q);", f"lw[0][j] = {helper}(pv + q);"),
            ("lw[1][j] = __ldg(pv + CHUNKS + q);",
             f"lw[1][j] = {helper}(pv + CHUNKS + q);"),
            ("cw[j] = __ldg(pv + 2 * CHUNKS + q);",
             f"cw[j] = {helper}(pv + 2 * CHUNKS + q);")):
        text = replace_pair_once(text, old, new)
    return text


def replace_x_loads(text: str, helper: str) -> str:
    for old, new in (
            ("xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));",
             f"xv[j][0] = {helper}(reinterpret_cast<const uint4*>(xp));"),
            ("xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));",
             f"xv[j][1] = {helper}(reinterpret_cast<const uint4*>(xp + 8));")):
        text = replace_pair_once(text, old, new)
    return text


def finish(version: int, label: str, text: str) -> None:
    old_ext = "okc_bf16proj_v262_c12pair_"
    if text.count(old_ext) != 1:
        raise RuntimeError("extension name marker mismatch")
    text = text.replace(old_ext, f"okc_bf16proj_v{version}_c12pair_", 1)
    text = text.replace("v262 c12 pairplanes", f"v{version} c12 pairplanes")
    output = ROOT / f"submission_plain_pairplanes_{label}_v{version}.py"
    compile(text, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(output.name, len(text.encode("utf-8")))


base = distribute_escape(BASE.read_text(encoding="utf-8"))
finish(265, "distributed_escape", base)

helpers_266 = r'''
__device__ __forceinline__ uint4 v266_pair_ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
'''
v266 = inject_helpers(base, helpers_266)
v266 = replace_weight_loads(v266, "v266_pair_ld4_na")
finish(266, "distributed_weightstream", v266)

helpers_267 = r'''
__device__ __forceinline__ uint64_t v267_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t v267_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v267_pair_ld4_ef(const uint4* p) {
  uint4 r; const uint64_t pol = v267_pair_pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 v267_pair_ld4_el(const uint4* p) {
  uint4 r; const uint64_t pol = v267_pair_pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
'''
v267 = inject_helpers(base, helpers_267)
v267 = replace_weight_loads(v267, "v267_pair_ld4_ef")
v267 = replace_x_loads(v267, "v267_pair_ld4_el")
finish(267, "distributed_cachehint", v267)
