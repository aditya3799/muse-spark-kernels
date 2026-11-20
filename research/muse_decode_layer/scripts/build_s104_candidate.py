"""Build submission S104:
1. Base: submission_s081_keell0renz_890809c.py
2. Retain full _mf_ab with MFAB_LAYER_REPS = 2, MFAB_LAYER_SEEDS = 1 (lightweight warmup, identical codegen)
3. In _make_chen:
   - Add small-m variants for m=16 (t16s5hzw, t16s4hzw, t16s5hze, t16s4hze)
   - Add ("qkv_gate", 16) and ("unpooler", 16) to RQBAKE_L
   - Guard _try(nm)
   - Add ('qkv', 16) and ('un', 16) to RQ_USE (leaving m=32 on the fast PDL sk2 path)
4. Append validated exact-C12 GEMV for T=1 from S073/S090.
5. Save with LF line endings, strictly < 1,000,000 bytes.
"""

import os
import re

# 1. Read base S081
with open('submission_s081_keell0renz_890809c.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 2. Modify _make_chen inside src:
# 2a. Add small-m variants into RQPROD
rqprod_old = 'RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",\n              "d2s5hzwP", "s4s5hzw"]'
rqprod_new = 'RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",\n              "d2s5hzwP", "s4s5hzw", "t16s5hzw", "t16s4hzw", "t16s5hze", "t16s4hze"]'
assert rqprod_old in src, "RQPROD not found"
src = src.replace(rqprod_old, rqprod_new, 1)

# 2b. Add m=16 into RQBAKE_L
rqbake_old = '''    RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }'''

rqbake_new = '''    RQBAKE_L = {
        ("qkv_gate", 16): ["t16s5hzw", "t16s4hzw"],
        ("unpooler", 16): ["t16s5hze", "t16s4hze"],
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }'''
assert rqbake_old in src, "RQBAKE_L not found"
src = src.replace(rqbake_old, rqbake_new, 1)

# 2c. Guard _try(nm)
try_old = '                def _try(nm):\n                    ksv = RQKSV.get(nm, KS)'
try_new = '                def _try(nm):\n                    if nm not in names: return None\n                    ksv = RQKSV.get(nm, KS)'
assert try_old in src, "_try(nm) not found"
src = src.replace(try_old, try_new, 1)

# 2d. Update RQ_USE to include ('qkv', 16) and ('un', 16)
rquse_old = "RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)}"
rquse_new = "RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128), ('qkv', 16), ('un', 16)}"
assert rquse_old in src, "RQ_USE not found"
src = src.replace(rquse_old, rquse_new, 1)

# 2e. Set MFAB reps/seeds for faster prepare while maintaining warmup
src = src.replace("MFAB_LAYER_SEEDS = 4\nMFAB_LAYER_REPS = 24", "MFAB_LAYER_SEEDS = 1\nMFAB_LAYER_REPS = 2", 1)

# 3. Read CGEMV from S090
with open('submission_s090_s081_cgemv_m1.py', 'r', encoding='utf-8') as f:
    s090 = f.read()

marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
appendix_pos = s090.rfind("# ---------------------------------------------------------------------------", 0, s090.index(marker))
cgemv_appendix = s090[appendix_pos:].rstrip()

# 4. In CGEMV appendix, ensure prior prepare chain links to s104
cgemv_appendix = cgemv_appendix.replace("_s073_prior_prepare = prepare", "_s104_prior_prepare = prepare\n_s073_prior_prepare = _s104_prior_prepare", 1)

# 5. Combine
final_src = src.rstrip() + "\n\n" + cgemv_appendix + "\n"

# Verify size and syntax
out_path = 'submission_s104_m16_cgemv.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(final_src)

size = os.path.getsize(out_path)
print(f"Built {out_path}: {size:,} bytes")
assert size < 1000000, f"File too large: {size} >= 1000000"
print(f"Headroom under 1,000,000: {1000000 - size:,} bytes")

# Python syntax check
compile(final_src, out_path, 'exec')
print("Syntax check: PASSED!")
