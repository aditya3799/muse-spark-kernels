"""Build a physically clean public N72 source with the legal BF16 unpooler split pipeline."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
base = (root / "submission_keell0renz_fair_base.py").read_text()
out = root / "submission_public_bf16_split_candidate.py"


marker = "\n\n# --------------------------------------------------------------------------------------\n# Weight repack: NVFP4 expert weights into the fused kernel's layout.\n"
helper = '''

def _mm_into(st, x, name, out, side=False):
    """The same BF16/FP32 GEMM as _mm, writing into a preallocated contiguous row slice."""
    T = x.shape[0]
    choice = st["lt_choice"].get((T, name)) if st.get("lt_choice") else None
    wn, f32 = LT_GEMMS[name]
    w = st[wn]
    if choice is None:
        if f32:
            out.copy_(torch.mm(x, w.t(), out_dtype=torch.float32))
        else:
            torch.mm(x, w.t(), out=out)
    elif choice[0] == "lt2":
        st["lt"].lt2_run(choice[1], x, w, st.get("lt2_wt", {}).get(wn, w), out,
                         st["lt_ws_side"] if side else st["lt_ws"])
    else:
        st["lt"].lt_run(choice[0], choice[1], x, w, out,
                        st["lt_ws_side"] if side else st["lt_ws"])
'''
assert base.count(marker) == 1 and "def _mm_into" not in base
base = base.replace(marker, helper + marker, 1)

old = '''    f = _mm(st, moe, "unpooler")
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    if recompute:
        st["mod"].boundary_rc(f, data["stream"], st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2,
                              hidden, 2, st["no_zero"], a, r1, st["a1"], st["b1"])
'''
new = '''    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    if T == 16384 and recompute:
        # Token rows are independent.  Run two legal BF16 8192-row GEMMs and let the side stream consume each
        # completed half through the unchanged MODE-2 boundary while the main stream produces the next half.
        f = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
        for lo, hi in ((0, 8192), (8192, T)):
            _mm_into(st, moe[lo:hi], "unpooler", f[lo:hi])
            side.wait_stream(main)
            with torch.cuda.stream(side):
                st["mod"].boundary_rc(f[lo:hi], data["stream"][lo:hi], st["g_post_ffn_norm"], st["a2"], st["b2"],
                                      st["g_next_pre_attn_norm"], stream2[lo:hi], hidden[lo:hi], 2, st["no_zero"],
                                      a[lo:hi], r1[lo:hi], st["a1"], st["b1"])
        main.wait_stream(side)
        return hidden, stream2, ids
    f = _mm(st, moe, "unpooler")
    if recompute:
        st["mod"].boundary_rc(f, data["stream"], st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2,
                              hidden, 2, st["no_zero"], a, r1, st["a1"], st["b1"])
'''
assert base.count(old) == 1
base = base.replace(old, new, 1)

for forbidden in (
    "ROUTE_SHAPING",
    "UNPOOL_FP8",
    "O_FP8",
    "torch._scaled_mm",
    "torch.ops.aten._scaled_mm",
    "base64.b64decode",
    "zlib.decompress",
):
    assert forbidden not in base, forbidden
ast.parse(base)
compile(base, str(out), "exec")
out.write_bytes(base.encode())
print(out.name, len(base.encode()))
