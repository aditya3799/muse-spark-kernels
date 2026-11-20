"""Generate v271: v248 pair-planes with one cache policy per thread.

Weights are one-pass and use L1 no-allocate/L2 evict-first.  The shared
activation uses L1/L2 evict-last in both the main vector loads and sparse
escape loads.  Escape-table entries use L1 no-allocate.  Unlike v256, policy
creation is hoisted out of every inlined load, so each thread executes only
two createpolicy instructions regardless of K/PER.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
OUTPUT = ROOT / "submission_plain_pairplanes_hoisted_cache_v271.py"
PAIR_MARKER = "# v106: exact C12 row-paired code planes"

text = BASE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


def replace_pair_once(old: str, new: str) -> None:
    global text
    head, sep, tail = text.partition(PAIR_MARKER)
    if not sep or tail.count(old) != 1:
        raise RuntimeError(
            f"expected one pair-block match, found {tail.count(old)}: "
            f"{old[:100]!r}")
    text = head + sep + tail.replace(old, new, 1)


replace_once(
    "_V106_CG_CUDA = r'''\n",
    """_V106_CG_CUDA = r'''
__device__ __forceinline__ uint64_t v271_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t v271_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
               : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v271_pair_ld4_ef(
    const uint4* p, uint64_t pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 v271_pair_ld4_el(
    const uint4* p, uint64_t pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ int2 v271_pair_ld2_na(const int2* p) {
  int2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned short v271_pair_ld1_el(
    const unsigned short* p, uint64_t pol) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.u16 %0, [%1], %2;"
               : "=h"(r) : "l"(p), "l"(pol));
  return r;
}
""",
)

replace_pair_once(
    "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n",
    "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n"
    "  const uint64_t weight_policy = v271_pair_pol_ef();\n"
    "  const uint64_t x_policy = v271_pair_pol_el();\n")

for old, new in (
        ("if (lane < pmax) mine = __ldg(&tb[lane]);",
         "if (lane < pmax) mine = v271_pair_ld2_na(&tb[lane]);"),
        ("if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);",
         "if (i0 > 0) en = (i <= cnt) ? v271_pair_ld2_na(&tb[i]) : make_int2(0, 0);"),
        ("const float c = __bfloat162float(x[en.x & 0xFFFF]) *\n"
         "                        esc_delta((uint32_t)en.y, base4 & 0xFFu);",
         "const unsigned short xb = v271_pair_ld1_el(\n"
         "            reinterpret_cast<const unsigned short*>(x) +\n"
         "            (en.x & 0xFFFF), x_policy);\n"
         "        const float c = __bfloat162float(__ushort_as_bfloat16(xb)) *\n"
         "                        esc_delta((uint32_t)en.y, base4 & 0xFFu);"),
        ("lw[0][j] = __ldg(pv + q);",
         "lw[0][j] = v271_pair_ld4_ef(pv + q, weight_policy);"),
        ("lw[1][j] = __ldg(pv + CHUNKS + q);",
         "lw[1][j] = v271_pair_ld4_ef(pv + CHUNKS + q, weight_policy);"),
        ("cw[j] = __ldg(pv + 2 * CHUNKS + q);",
         "cw[j] = v271_pair_ld4_ef(pv + 2 * CHUNKS + q, weight_policy);"),
        ("xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));",
         "xv[j][0] = v271_pair_ld4_el(\n"
         "        reinterpret_cast<const uint4*>(xp), x_policy);"),
        ("xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));",
         "xv[j][1] = v271_pair_ld4_el(\n"
         "        reinterpret_cast<const uint4*>(xp + 8), x_policy);")):
    replace_pair_once(old, new)

replace_once("okc_bf16proj_v248_c12pair_",
             "okc_bf16proj_v271_c12pair_")
text = text.replace("v248 c12 pairplanes", "v271 c12 pairplanes")

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
