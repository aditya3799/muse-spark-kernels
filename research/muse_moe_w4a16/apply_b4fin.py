import py_compile
import shutil

# Restore clean copy
shutil.copyfile("submission_pre_b4fin.py", "submission.py")

with open("submission.py") as f:
    s = f.read()

idx_main = s.find("ALT_SRCS['v167_main'] = '")
assert idx_main != -1

s_pre = s[:idx_main]
s_main = s[idx_main:]

target1 = r"\n#define OKC_MFQ 47   // 1 doneAs parity, 2 xready parity, 4 no grid barrier, 8 launch parity / zeroing, 32 volatile S.gen loads\n#endif"
repl1 = target1 + r"\n#ifndef OKC_B4FIN\n#define OKC_B4FIN 0\n#endif\n#ifndef B4_EARLY_Y\n#define B4_EARLY_Y 1\n#endif"

target2 = r"  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};"
repl2 = r"  __nv_bfloat16* out;\n  unsigned long long* dbg;\n  float* sspart;\n};"

target3 = r"        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n#endif\n      }\n    }\n  }\n}"
repl3 = r"""        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n#endif\n      }\n    }\n#if OKC_B4FIN\n    {\n      float q0 = 0.f, q1 = 0.f;\n      if (tok0 < cnt) { const float a = d4[0] * w0, b = d4[2] * w1; q0 = a * a + b * b; }\n      if (tok0 + 1 < cnt) { const float a = d4[1] * w0, b = d4[3] * w1; q1 = a * a + b * b; }\n#pragma unroll\n      for (int o = 4; o < 32; o <<= 1) {\n        q0 += __shfl_xor_sync(0xffffffffu, q0, o);\n        q1 += __shfl_xor_sync(0xffffffffu, q1, o);\n      }\n      if (g == 0) {\n        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;\n        if (tok0 + 1 < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;\n      }\n    }\n#endif\n  }\n}"""

target4 = r"  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_WGSFINSS"
repl4 = r"""  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_B4FIN\n  {\n    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");\n    float* red = reinterpret_cast<float*>(xbuf);\n    const int lane = tid & 31, warp = WARP_IDX(tid);\n    constexpr int SEG = 1024, NSEG = HD / SEG;\n    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {\n      const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;\n#if OKC_MFIN && (OKC_MFP & 4)\n      if (tid < 8) {\n        if (S.pos[tk * 8 + tid] >= 0) {\n          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];\n          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);\n          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;\n          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }\n        }\n      }\n      CTA_SYNC();\n#endif\n#if B4_EARLY_Y\n      float2 yv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        const int q = S.pos[tk * 8 + sl];\n        yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);\n      }\n#endif\n      {\n        const int sl = warp >> 1, hh = warp & 1;\n        const int q = S.pos[tk * 8 + sl];\n        float v = 0.f;\n        if (q >= 0) {\n          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));\n          v = (a.x + a.y) + (a.z + a.w);\n        }\n#pragma unroll\n        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n        if (lane == 0) red[warp] = v;\n      }\n      CTA_SYNC();\n      float2 acc = make_float2(0.f, 0.f);\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        const int q = S.pos[tk * 8 + sl];\n        if (q >= 0) {\n          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);\n#if B4_EARLY_Y\n          const float2 v = yv[sl];\n#else\n          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));\n#endif\n          acc.x += r * v.x; acc.y += r * v.y;\n        }\n      }\n      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\n      CTA_SYNC();\n    }\n  }\n#elif OKC_WGS && OKC_WGSFINSS"""

target5 = r"  p.ybuf = p.ss + okc::EMAX;\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();"
repl5 = r"  p.ybuf = p.ss + okc::EMAX;\n#if OKC_WGS && OKC_B4FIN\n  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, \"B4FIN build is for M <= 32\");\n  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;\n#else\n  p.sspart = nullptr;\n#endif\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();"

for t, r in [(target1, repl1), (target2, repl2), (target3, repl3), (target4, repl4), (target5, repl5)]:
    assert s_main.count(t) == 1, f"Count in s_main of {t[:40]} is {s_main.count(t)}"
    s_main = s_main.replace(t, r, 1)

# Now add -DOKC_B4FIN=1 to V167_M32_DEFINES (which is in s_pre or s_main)
s = s_pre + s_main
old_m32 = "'-DOKC_WPBURST=3')"
new_m32 = "'-DOKC_WPBURST=3', '-DOKC_B4FIN=1')"
assert s.count(old_m32) == 1, f"old_m32 count is {s.count(old_m32)}"
s = s.replace(old_m32, new_m32)

with open("submission.py", "w") as f:
    f.write(s)

py_compile.compile("submission.py", doraise=True)
print("SUCCESS: submission.py patched with B4FIN and compiled cleanly!")
