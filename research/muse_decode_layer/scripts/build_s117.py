with open('submission_s114_rq_qkv_fixed2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update RQBAKE_L to add unpooler 16 and 32
old_rqbake = '''        ("qkv_gate", 32): ["t32s6hzw", "t32s5hzw"],
        ("qkv_gate", 16): ["t16s5hzw", "t16s4hzw"],'''

new_rqbake = '''        ("qkv_gate", 32): ["t32s6hzw", "t32s5hzw"],
        ("qkv_gate", 16): ["t16s5hzw", "t16s4hzw"],
        ("unpooler", 32): ["t32s6hzw", "t32s5hzw"],
        ("unpooler", 16): ["t16s5hzw", "t16s4hzw"],'''

assert old_rqbake in content, "old_rqbake anchor not found"
content = content.replace(old_rqbake, new_rqbake, 1)

# 2. Allow m in (16, 32, 128, 256) for unpooler in _rqbake
old_loop1 = 'for m in ((16, 32, 128, 256) if layer == "qkv_gate" else (128, 256)):'
new_loop1 = 'for m in (16, 32, 128, 256):'
assert content.count(old_loop1) == 2, f"Expected 2 occurrences of old_loop1, found {content.count(old_loop1)}"
content = content.replace(old_loop1, new_loop1)

# 3. Add un 16 and 32 to RQ_USE
old_rquse = "RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128), ('qkv', 32), ('qkv', 16)}"
new_rquse = "RQ_USE = {('un', 16), ('un', 32), ('un', 128), ('qkv', 256), ('qkv', 128), ('qkv', 32), ('qkv', 16)}"
assert old_rquse in content, "old_rquse anchor not found"
content = content.replace(old_rquse, new_rquse, 1)

# 4. Dispatch _rq_gemm in dense
old_dense = '''def dense(x, w, T, state=None, name=None):
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)'''

new_dense = '''def dense(x, w, T, state=None, name=None):
    if state is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)'''

assert old_dense in content, "old_dense anchor not found"
content = content.replace(old_dense, new_dense, 1)

# 5. Update build name for inline C++ extension
content = content.replace('muse_att_v298w1pf_s114_', 'muse_att_v298w1pf_s117_')

# Convert strictly to Unix LF (\n) line endings
content_bytes = content.encode('utf-8').replace(b'\r\n', b'\n')
print("Size of s117:", len(content_bytes), "bytes")
assert len(content_bytes) < 1000000, "File too large!"

with open('submission_s117_rq_unpooler.py', 'wb') as f:
    f.write(content_bytes)

print("Created submission_s117_rq_unpooler.py successfully!")
