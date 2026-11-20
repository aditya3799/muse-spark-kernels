"""Generate v256: apply SM90 cache hints directly to v248 pair-planes.

Pair-plane weights are a one-pass stream, while the 2--8 KiB activation is
shared by every output CTA.  This variant keeps v248's packing, arithmetic,
reduction order, validation, and per-route gate, changing only the three
16-byte weight loads to evict-first/no-allocate and the two activation loads
to evict-last.  O and Unpooler are still promoted independently.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
OUTPUT = ROOT / "submission_plain_pairplanes_cachehint_v256.py"

text = BASE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


def replace_pair_once(old: str, new: str) -> None:
    """Replace only inside v248's appended pair-plane block."""
    global text
    marker = "# v106: exact C12 row-paired code planes"
    head, sep, tail = text.partition(marker)
    if not sep or tail.count(old) != 1:
        raise RuntimeError(
            f"expected one pair-block match, found {tail.count(old)}: "
            f"{old[:100]!r}")
    text = head + sep + tail.replace(old, new, 1)


replace_once(
    "_V106_CG_CUDA = r'''\n",
    """_V106_CG_CUDA = r'''
__device__ __forceinline__ uint64_t v256_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t v256_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v256_pair_ld4_ef(const uint4* p) {
  uint4 r; const uint64_t pol = v256_pair_pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 v256_pair_ld4_el(const uint4* p) {
  uint4 r; const uint64_t pol = v256_pair_pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
""",
)

for old, new in (
        ("lw[0][j] = __ldg(pv + q);",
         "lw[0][j] = v256_pair_ld4_ef(pv + q);"),
        ("lw[1][j] = __ldg(pv + CHUNKS + q);",
         "lw[1][j] = v256_pair_ld4_ef(pv + CHUNKS + q);"),
        ("cw[j] = __ldg(pv + 2 * CHUNKS + q);",
         "cw[j] = v256_pair_ld4_ef(pv + 2 * CHUNKS + q);"),
        ("xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));",
         "xv[j][0] = v256_pair_ld4_el(reinterpret_cast<const uint4*>(xp));"),
        ("xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));",
         "xv[j][1] = v256_pair_ld4_el(reinterpret_cast<const uint4*>(xp + 8));")):
    replace_pair_once(old, new)

replace_once("okc_bf16proj_v248_c12pair_",
             "okc_bf16proj_v256_c12pair_")
text = text.replace("v248 c12 pairplanes", "v256 c12 pairplanes")

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
