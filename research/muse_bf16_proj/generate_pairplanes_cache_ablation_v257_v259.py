"""Generate three cache-policy ablations for v248 pair-plane O/Unpooler.

v257 streams pair-plane weights through L1 no-allocate with ordinary x loads.
v258 also marks those weight lines evict-first in L2.  v259 leaves weights on
the incumbent path and marks only the shared activation evict-last in L1/L2.
Packing, arithmetic, validation, and v248's independent one-percent gates are
unchanged.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
PAIR_MARKER = "# v106: exact C12 row-paired code planes"


def replace_pair_once(text: str, old: str, new: str) -> str:
    head, sep, tail = text.partition(PAIR_MARKER)
    if not sep or tail.count(old) != 1:
        raise RuntimeError(
            f"expected one pair-block match, found {tail.count(old)}: "
            f"{old[:100]!r}")
    return head + sep + tail.replace(old, new, 1)


def inject_helpers(text: str, helpers: str) -> str:
    anchor = "_V106_CG_CUDA = r'''\n"
    if text.count(anchor) != 1:
        raise RuntimeError("pair CUDA marker mismatch")
    return text.replace(anchor, anchor + helpers.strip() + "\n", 1)


def replace_weight_loads(text: str, helper: str) -> str:
    for old, new in (
            ("lw[0][j] = __ldg(pv + q);",
             f"lw[0][j] = {helper}(pv + q);"),
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
    old_ext = "okc_bf16proj_v248_c12pair_"
    if text.count(old_ext) != 1:
        raise RuntimeError("extension name marker mismatch")
    text = text.replace(old_ext, f"okc_bf16proj_v{version}_c12pair_", 1)
    text = text.replace("v248 c12 pairplanes",
                        f"v{version} c12 pairplanes")
    output = ROOT / f"submission_plain_pairplanes_{label}_v{version}.py"
    compile(text, str(output), "exec")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(output.name, len(text.encode("utf-8")))


base = BASE.read_text(encoding="utf-8")

helpers_257 = r'''
__device__ __forceinline__ uint4 v257_pair_ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
'''
v257 = inject_helpers(base, helpers_257)
v257 = replace_weight_loads(v257, "v257_pair_ld4_na")
finish(257, "weight_stream", v257)

helpers_258 = r'''
__device__ __forceinline__ uint64_t v258_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v258_pair_ld4_ef(const uint4* p) {
  uint4 r; const uint64_t pol = v258_pair_pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
'''
v258 = inject_helpers(base, helpers_258)
v258 = replace_weight_loads(v258, "v258_pair_ld4_ef")
finish(258, "weight_l2ef", v258)

helpers_259 = r'''
__device__ __forceinline__ uint64_t v259_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v259_pair_ld4_el(const uint4* p) {
  uint4 r; const uint64_t pol = v259_pair_pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
'''
v259 = inject_helpers(base, helpers_259)
v259 = replace_x_loads(v259, "v259_pair_ld4_el")
finish(259, "x_resident", v259)
