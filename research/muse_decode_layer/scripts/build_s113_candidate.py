with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update module name for attention extension
text = text.replace('muse_att_v298w1pf_s108_{arch}', 'muse_att_v298w1pf_s113_{arch}')

# 2. Add t32 and t16 to RQPROD
old_rqprod = 'RQPROD = ["s4s5z", "s4s5hz", "s4s5hze", "d2s5hze", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]'
assert old_rqprod in text, "old_rqprod not found"
new_rqprod = 'RQPROD = ["s4s5z", "s4s5hz", "s4s5hze", "d2s5hze", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH", "t32s6hzw", "t32s5hzw", "t16s5hzw", "t16s4hzw"]'
text = text.replace(old_rqprod, new_rqprod, 1)

# 3. Add to RQBAKE_L
old_rqbake_l = 'RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)'
assert old_rqbake_l in text, "old_rqbake_l not found"
new_rqbake_l = '''RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        ("qkv_gate", 32): ["t32s6hzw", "t32s5hzw"],
        ("qkv_gate", 16): ["t16s5hzw", "t16s4hzw"],'''
text = text.replace(old_rqbake_l, new_rqbake_l, 1)

# 4. Update RQBAKE
old_rqbake = 'RQBAKE = {128: [], 256: []}'
assert old_rqbake in text, "old_rqbake not found"
new_rqbake = 'RQBAKE = {16: [], 32: [], 128: [], 256: []}'
text = text.replace(old_rqbake, new_rqbake, 1)

# 5. In _rqbake: allow m in (16, 32, 128, 256) for qkv_gate (first loop)
old_bake_m = 'for m in (128, 256):  # r6 timed before the pack (critic F1: the first r6 timing after the pack read ~4 us high)'
assert old_bake_m in text, "old_bake_m not found"
new_bake_m = 'for m in ((16, 32, 128, 256) if layer == "qkv_gate" else (128, 256)):'
text = text.replace(old_bake_m, new_bake_m, 1)

# 6. In _rqbake candidates loop:
old_bl = '_bl = RQBAKE_L.get((layer, 128), RQBAKE[128]) + RQBAKE_L.get((layer, 256), RQBAKE[256])'
assert old_bl in text, "old_bl not found"
new_bl = '_bl = RQBAKE_L.get((layer, 16), []) + RQBAKE_L.get((layer, 32), []) + RQBAKE_L.get((layer, 128), RQBAKE.get(128, [])) + RQBAKE_L.get((layer, 256), RQBAKE.get(256, []))'
text = text.replace(old_bl, new_bl, 1)

# 7. In _rqbake second loop over m:
idx_first = text.find(new_bake_m)
idx_second = text.find('for m in (128, 256):', idx_first)
assert idx_second != -1, "second for m in (128, 256) not found"
text = text[:idx_second] + 'for m in ((16, 32, 128, 256) if layer == "qkv_gate" else (128, 256)):' + text[idx_second + len('for m in (128, 256):'):]

# 8. In _rqbake: safe names_m lookup
old_names_m = 'names_m = RQBAKE_L.get((layer, m), RQBAKE[m])'
assert old_names_m in text, "old_names_m not found"
new_names_m = 'names_m = RQBAKE_L.get((layer, m), RQBAKE.get(m, []))'
text = text.replace(old_names_m, new_names_m, 1)

# 9. Update RQ_USE
old_rquse = "RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)}"
assert old_rquse in text, "old_rquse not found"
new_rquse = "RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128), ('qkv', 32), ('qkv', 16)}"
text = text.replace(old_rquse, new_rquse, 1)

# 10. Update kernel_fn to dispatch _rq_gemm for T in (16, 32)
old_kfn = '''def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None
    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is None:
            qkv = _graft_lt_run(data["attn_input"], plan)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    return _vj_kernel_fn(data, state)'''

new_kfn = '''def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    if T in (16, 32) and state.get("att_ok") and _supported7(cfg) and _att_fits7(data, cfg):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is not None:
            try:
                o_in = _decode_attention7(data, state, qkv)
            except RuntimeError:
                return _kernel_fn_fa3(data, state)
            return _after_attention(data, state, o_in, None, T)
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None
    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is None:
            qkv = _graft_lt_run(data["attn_input"], plan)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    return _vj_kernel_fn(data, state)'''

assert old_kfn in text, "old_kfn not found"
text = text.replace(old_kfn, new_kfn, 1)

out_file = 'submission_s113_rq_qkv_fixed.py'
with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

import os
size = os.path.getsize(out_file)
print(f"Successfully generated {out_file}, size = {size} bytes")
assert size < 1000000, f"Size {size} exceeds 1,000,000 bytes!"
