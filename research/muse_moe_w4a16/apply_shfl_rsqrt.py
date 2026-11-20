with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
assert idx_main != -1

target = r"""      float2 acc = make_float2(0.f, 0.f);\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        const int q = S.pos[tk * 8 + sl];\n        if (q >= 0) {\n          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);\n#if B4_EARLY_Y\n          const float2 v = yv[sl];\n#else\n          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));\n#endif\n          acc.x += r * v.x; acc.y += r * v.y;\n        }\n      }"""

replacement = r"""      float r_lane = 0.f;\n      if (lane < 8) {\n        const int q = S.pos[tk * 8 + lane];\n        if (q >= 0) {\n          r_lane = p.topk_w[tk * 8 + lane] * rsqrtf((red[2 * lane] + red[2 * lane + 1]) * (1.f / HD) + 1e-8f);\n        }\n      }\n      float2 acc = make_float2(0.f, 0.f);\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        const int q = S.pos[tk * 8 + sl];\n        if (q >= 0) {\n          const float r = __shfl_sync(0xffffffffu, r_lane, sl);\n#if B4_EARLY_Y\n          const float2 v = yv[sl];\n#else\n          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));\n#endif\n          acc.x += r * v.x; acc.y += r * v.y;\n        }\n      }"""

assert target in text[idx_main:], "Target string not found in v167_main!"

text_new = text[:idx_main] + text[idx_main:].replace(target, replacement, 1)

with open('submission.py', 'w', encoding='utf-8') as f:
    f.write(text_new)

print("Successfully replaced rsqrt loop with parallel lane shuffle in submission.py!")
