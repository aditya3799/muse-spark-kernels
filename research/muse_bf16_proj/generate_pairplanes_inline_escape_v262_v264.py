"""Generate pair-plane variants without the dedicated escape helper warp.

Warp 0 performs the same fixed-order escape correction before its ordinary
weight chunk.  Other worker warps can start immediately, and the existing
final CTA barrier makes the helper slot visible.  This removes one 32-thread
warp from every CTA without changing packing, FMA order, reduction order, or
the correction sum.  v262 changes topology only; v263 adds L1-no-allocate
weight streaming; v264 adds the full weight-evict-first/x-evict-last policy.
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


def replace_pair_all(text: str, old: str, new: str, count: int) -> str:
    head, sep, tail = text.partition(PAIR_MARKER)
    if not sep or tail.count(old) != count:
        raise RuntimeError(
            f"expected {count} pair-block matches, found {tail.count(old)}: "
            f"{old[:100]!r}")
    return head + sep + tail.replace(old, new)


def inline_escape(text: str) -> str:
    text = replace_pair_once(
        text, "__global__ void __launch_bounds__(NTH + 32) cgemv_pair_kernel(",
        "__global__ void __launch_bounds__(NTH) cgemv_pair_kernel(")
    text = replace_pair_once(text, "  if (warp == NWARP) {",
                             "  if (warp == 0) {")
    text = replace_pair_once(
        text,
        "    __syncthreads();\n    return;\n  }\n\n  uint4 lw[ROWS][PER];",
        "    __syncwarp();\n  }\n\n  uint4 lw[ROWS][PER];")
    text = replace_pair_once(
        text, "&active, kern, NTH + 32, 0));",
        "&active, kern, NTH, 0));")
    for old, new, count in (
            ("<<<N / 2, 288, 0, st>>>", "<<<N / 2, 256, 0, st>>>", 2),
            ("<<<N / 2, 544, 0, st>>>", "<<<N / 2, 512, 0, st>>>", 1),
            ("<<<N / 2, 64, 0, st>>>", "<<<N / 2, 32, 0, st>>>", 1),
            ("<<<N / 2, 96, 0, st>>>", "<<<N / 2, 64, 0, st>>>", 1),
            ("<<<N / 2, 160, 0, st>>>", "<<<N / 2, 128, 0, st>>>", 1)):
        text = replace_pair_all(text, old, new, count)
    return text


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


base = inline_escape(BASE.read_text(encoding="utf-8"))
finish(262, "inline_escape", base)

helpers_263 = r'''
__device__ __forceinline__ uint4 v263_pair_ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
'''
v263 = inject_helpers(base, helpers_263)
v263 = replace_weight_loads(v263, "v263_pair_ld4_na")
finish(263, "inline_weightstream", v263)

helpers_264 = r'''
__device__ __forceinline__ uint64_t v264_pair_pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t v264_pair_pol_el() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 v264_pair_ld4_ef(const uint4* p) {
  uint4 r; const uint64_t pol = v264_pair_pol_ef();
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 v264_pair_ld4_el(const uint4* p) {
  uint4 r; const uint64_t pol = v264_pair_pol_el();
  asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
'''
v264 = inject_helpers(base, helpers_264)
v264 = replace_weight_loads(v264, "v264_pair_ld4_ef")
v264 = replace_x_loads(v264, "v264_pair_ld4_el")
finish(264, "inline_cachehint", v264)
