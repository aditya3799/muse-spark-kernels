from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_bulk_l264_exact_wst.py"
TARGET = ROOT / "submission_bulk_l264_poly3_wst.py"

text = SOURCE.read_text()

old_helper = r'''// OKC: exact exp2 on a packed f16 pair (one MUFU per 2 scores) and the f16x2 -> e4m3x2 conversion of the P codes
DINL unsigned ex2_f16x2(unsigned x) {
    unsigned r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}'''

new_helper = r'''// OKC: packed-half degree-3 exp2 for WST.  The polynomial is the same bounded-error
// approximation used by the incumbent's float path (relative error about 9e-5 before f16
// rounding).  Clamp masked/underflowed lanes to 2^-14, build the exact power-of-two scale
// in the f16 exponent bits, and retain the normal f16x2 -> E4M3 RN conversion below.
// This attends every key and is substantially more accurate than the disabled Schraudolph path.
DINL unsigned ex2_poly_f16x2(unsigned xb) {
    const __half2 hmin = __float2half2_rn(-14.0f);
    const __half2 x = __hmax2(u32h2(xb), hmin);
    const __half2 j = h2floor(x);
    const __half2 f = __hsub2(x, j);
    __half2 p = __hfma2(f, __float2half2_rn(0.0790209f), __float2half2_rn(0.2249424f));
    p = __hfma2(f, p, __float2half2_rn(0.6960656f));
    p = __hfma2(f, p, __float2half2_rn(0.9999893f));
    const int jl = __half2int_rz(__low2half(j));
    const int jh = __half2int_rz(__high2half(j));
    const unsigned pb = h2u32(p);
    const unsigned lo = ((pb & 0xffffu) + (unsigned)(jl * 1024)) & 0xffffu;
    const unsigned hi = (((pb >> 16) + (unsigned)(jh * 1024)) & 0xffffu) << 16;
    return lo | hi;
}'''

old_call = "s[j] = ex2_f16x2(h2u32(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1])));"
new_call = "s[j] = ex2_poly_f16x2(h2u32(__hfma2(u32h2(s[j]), qk8h2, kc[j & 1])));"

if text.count(old_helper) != 1:
    raise SystemExit(f"expected one exact helper, found {text.count(old_helper)}")
if text.count(old_call) != 1:
    raise SystemExit(f"expected one exact call, found {text.count(old_call)}")

text = text.replace(old_helper, new_helper).replace(old_call, new_call)
TARGET.write_text(text)
print(TARGET)
