with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx1 = text.find('_MA_ALT_SRCS["moe_helper0"] = ')
idx2 = text.find('\nMOE_MID_DEFINES +=', idx1)
moe_helper0_code = text[idx1:idx2]
locs = {'_MA_ALT_SRCS': {}}
exec(moe_helper0_code, {}, locs)
helper0_src = locs['_MA_ALT_SRCS']['moe_helper0']

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

edits = [
    (e0_old, e0_new),
    (e1_old, e1_new),
    (e2_old, e2_new),
    (e3_old, e3_new),
    (e4_old, e4_new),
    (e5_old, e5_new),
]

s = helper0_src
for i, (old, new) in enumerate(edits):
    c = s.count(old)
    print(f"Edit {i}: count={c}")
    assert c == 1, f"Edit {i} failed: {c}"
    s = s.replace(old, new)

print("All 6 minimal edits replaced cleanly!")
