with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx1 = text.find('_MA_ALT_SRCS["moe_helper0"] = ')
idx2 = text.find('\nMOE_MID_DEFINES +=', idx1)
moe_helper0_code = text[idx1:idx2]
locs = {'_MA_ALT_SRCS': {}}
exec(moe_helper0_code, {}, locs)
helper0_src = locs['_MA_ALT_SRCS']['moe_helper0']

with open('../inference_research/keell0renz_muse/kernels/muse_moe_w4a16.py', 'r', encoding='utf-8') as f:
    k_text = f.read()
start = k_text.find('PS_PATCHES = ')
end = k_text.find('\nfor _fn,', start)
ps_locs = {}
exec(k_text[start:end], {}, ps_locs)
edits = list(ps_locs['PS_PATCHES']['t2_wsi'][1])

# Adapt Edit 0 for helper0
ed0_old = """
constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
"""

ed0_new = """
constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
#ifndef OKC_WSI
#define OKC_WSI 1   // wsi: dedicated ring-issuer warp in the skinny WS builds (make_wsi.py)
#endif
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_WSI ? 64 : 32) : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
"""
edits[0] = (ed0_old, ed0_new)

print('Testing 6 edits against _MA_ALT_SRCS["moe_helper0"]:')
s = helper0_src
for i, (old, new) in enumerate(edits):
    c = s.count(old)
    print(f'  Edit {i}: count={c}')
    assert c == 1, f"Edit {i} failed: count={c}"
    s = s.replace(old, new)
print('SUCCESS! All 6 edits matched uniquely and replaced cleanly.')
