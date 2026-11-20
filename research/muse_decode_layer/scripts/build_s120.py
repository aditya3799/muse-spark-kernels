with open('submission_s114_rq_qkv_fixed2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Attention q8-w0 / fpart optimization
s1_old = "const bool w0on = OKC_W0ENABLE && p.coop && p.resident &&"
s1_new = "const bool w0on = OKC_W0ENABLE && L == 8 && p.coop && p.resident &&"
assert content.count(s1_old) == 1, "s1_old anchor not unique"
content = content.replace(s1_old, s1_new)

s2_old = "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 &&"
s2_new = "constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0L1 &&"
assert content.count(s2_old) == 1, "s2_old anchor not unique"
content = content.replace(s2_old, s2_new)

# 2. MoE t2_wsi ring-issuer warp for M=8 (verify 1)
e0_old = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);"
e0_new = "#ifndef OKC_WSI\n#define OKC_WSI 1\n#endif\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_WSI ? 64 : 32) : 0);"

e1_old = "    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB:"
e1_new = "#if OKC_WSI\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wse + sl);\n#endif\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    // WSB:"

e2_old = "if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);"
e2_new = "if (!OKC_WSI && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);"

e3_old = "if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);"
e3_new = "if (!OKC_WSI && lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);"

e4_old = "  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n#endif\n\n"
e4_new = """  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

#if OKC_WS && OKC_WSI
__device__ void ws_issuer(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  if (lane == 0) {
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
    for (int u = OKC_NS; u < N; ++u) {
      const int q = u - OKC_NS;
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, u, nA, a0, b0, ring);
    }
  }
  __syncwarp();
  WS_JOIN();
}
#endif

"""

e5_old = "#if OKC_WS\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }"
e5_new = "#if OKC_WS\n#if OKC_WSI\n  if (tid >= NT + 32) { ws_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }"

old_tiny2_decl = 'MOE_TINY2_DEFINES += ("@moe_helper0",)'
assert content.count(old_tiny2_decl) == 1, "old_tiny2_decl anchor not unique"

wsi_injection = f'''def _t2_wsi_patch(c):
    for old, new in [
        ({repr(e0_old)}, {repr(e0_new)}),
        ({repr(e1_old)}, {repr(e1_new)}),
        ({repr(e2_old)}, {repr(e2_new)}),
        ({repr(e3_old)}, {repr(e3_new)}),
        ({repr(e4_old)}, {repr(e4_new)}),
        ({repr(e5_old)}, {repr(e5_new)}),
    ]:
        assert c.count(old) == 1, f"t2_wsi anchor not found: {{old[:60]!r}}"
        c = c.replace(old, new)
    return c

_MA_ALT_SRCS["t2_wsi"] = _t2_wsi_patch(_MA_ALT_SRCS["moe_helper0"])
MOE_TINY2_DEFINES = tuple(d for d in MOE_TINY2_DEFINES if not d.startswith("@")) + ("@t2_wsi",)
'''

content = content.replace(old_tiny2_decl, wsi_injection)

# 3. Update attention inline extension module name
old_mod = 'muse_att_v298w1pf_s114_'
new_mod = 'muse_att_v298w1pf_s120_'
assert content.count(old_mod) == 1, "old_mod anchor not unique"
content = content.replace(old_mod, new_mod)

# Encode with strict Unix LF
b_content = content.encode('utf-8').replace(b'\r\n', b'\n')
print(f"Generated S120 byte size: {len(b_content)}")
assert len(b_content) < 1000000, f"File too large: {len(b_content)} >= 1000000"

with open('submission_s120_wsi_fpart.py', 'wb') as f:
    f.write(b_content)

print("Successfully written submission_s120_wsi_fpart.py")
