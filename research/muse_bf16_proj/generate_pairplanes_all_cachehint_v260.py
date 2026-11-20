"""Generate v260: cache-hinted pair-planes across all four m=1 dense routes.

Unlike v248/v256, this discovery isolate also retests QKV and Pooler because
their pair-plane packing may interact with the cache policy.  Every route is
packed losslessly, exhaustively validated, compared directly with complete
v221, and promoted independently only on a one-percent adjusted-core win.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_c12_pairplanes_v106.py"
OUTPUT = ROOT / "submission_plain_pairplanes_all_cachehint_v260.py"

text = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
marker = "# v106: exact C12 row-paired code planes for the scored m=1 dense routes."
if donor.count(marker) != 1:
    raise RuntimeError("v106 pair-plane marker mismatch")
block = marker + donor.split(marker, 1)[1]

old_cuda = "_V106_CG_CUDA = r'''\n"
helpers = r'''
__device__ __forceinline__ uint64_t v260_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t v260_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v260_pair_ld4_ef(const uint4* p) {
  uint4 r; const uint64_t pol = v260_pair_pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 v260_pair_ld4_el(const uint4* p) {
  uint4 r; const uint64_t pol = v260_pair_pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
'''
if block.count(old_cuda) != 1:
    raise RuntimeError("v106 pair CUDA marker mismatch")
block = block.replace(old_cuda, old_cuda + helpers.strip() + "\n", 1)

for old, new in (
        ("lw[0][j] = __ldg(pv + q);",
         "lw[0][j] = v260_pair_ld4_ef(pv + q);"),
        ("lw[1][j] = __ldg(pv + CHUNKS + q);",
         "lw[1][j] = v260_pair_ld4_ef(pv + CHUNKS + q);"),
        ("cw[j] = __ldg(pv + 2 * CHUNKS + q);",
         "cw[j] = v260_pair_ld4_ef(pv + 2 * CHUNKS + q);"),
        ("xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));",
         "xv[j][0] = v260_pair_ld4_el(reinterpret_cast<const uint4*>(xp));"),
        ("xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));",
         "xv[j][1] = v260_pair_ld4_el(reinterpret_cast<const uint4*>(xp + 8));")):
    if block.count(old) != 1:
        raise RuntimeError("v106 pair load marker mismatch: " + old)
    block = block.replace(old, new, 1)

block = block.replace("_V106_REQUIRED_CORE_RATIO = 0.97",
                      "_V106_REQUIRED_CORE_RATIO = 0.99", 1)
block = block.replace("okc_bf16proj_v106_c12pair_",
                      "okc_bf16proj_v260_c12pair_", 1)
block = block.replace("v106 c12 pairplanes", "v260 c12 pairplanes")

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
