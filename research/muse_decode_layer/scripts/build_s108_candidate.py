"""Build submission S108:
1. Base: submission_s081_keell0renz_890809c.py
2. Retain OKC_NP = 128 (exact proven value from S081/S085)
3. Add S085 q1_occ3 occupancy plan:
   - #define OKC_WSMINB1 3
   - MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2)
   - Launch ft = ((3 * sms) if L == 1 else target)
4. Add S083 L2::64B copy hint for flat page-1 cache copies
5. Unique module name: muse_att_v298w1pf_s108_{arch}
6. Append validated exact-C12 GEMV for T=1 from S073/S090.
7. Save with LF line endings, strictly < 1,000,000 bytes.
"""

import os
import re

# 1. Read base S081
with open('submission_s081_keell0renz_890809c.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 2. Add L2::64B copy hint
old_helper = '''__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
'''
new_helper = old_helper + '''__device__ __forceinline__ void cp_async16_l2_64(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}
'''
assert src.count(old_helper) == 1, "old_helper match error"
src = src.replace(old_helper, new_helper, 1)

old_calls = '''            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
'''
new_calls = '''            cp_async16_l2_64(kb + pso + k * (PR * ROWB), p.k_cache + off);
            cp_async16_l2_64(vb + pso + k * (PR * ROWB), p.v_cache + off);
'''
assert src.count(old_calls) == 1, "old_calls match error"
src = src.replace(old_calls, new_calls, 1)

# 3. Add q1_occ3 attention topology (keeping NP = 128)
wsminb_old = "#define OKC_WSMINB 2\n"
wsminb_new = "#define OKC_WSMINB 2\n#define OKC_WSMINB1 3   // q_len 1 flat: 3 resident CTAs per SM\n"
assert wsminb_old in src, "OKC_WSMINB not found"
src = src.replace(wsminb_old, wsminb_new, 1)

minb_old = "static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);"
minb_new = "static constexpr int MINB = WS ? (F2 ? 1 : ((L == 1) ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);"
assert minb_old in src, "MINB not found"
src = src.replace(minb_old, minb_new, 1)

ft_old = "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target"
ft_new = "ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)"
assert ft_old in src, "ft line not found"
src = src.replace(ft_old, ft_new, 1)

mod_old = 'load_inline(name=f"muse_att_v298w1pf_{arch}"'
mod_new = 'load_inline(name=f"muse_att_v298w1pf_s108_{arch}"'
assert mod_old in src, "mod name not found"
src = src.replace(mod_old, mod_new, 1)

# 4. Append exact-C12 GEMV from S090
with open('submission_s090_s081_cgemv_m1.py', 'r', encoding='utf-8') as f:
    s090 = f.read()

marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
appendix_pos = s090.rfind("# ---------------------------------------------------------------------------", 0, s090.index(marker))
cgemv_appendix = s090[appendix_pos:].rstrip()

# Link prepare chain to s108
cgemv_appendix = cgemv_appendix.replace("_s073_prior_prepare = prepare", "_s108_prior_prepare = prepare\n_s073_prior_prepare = _s108_prior_prepare", 1)

# 5. Combine
final_src = src.rstrip() + "\n\n" + cgemv_appendix + "\n"

# Verify size and syntax
out_path = 'submission_s108_att_cgemv.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(final_src)

size = os.path.getsize(out_path)
print(f"Built {out_path}: {size:,} bytes")
assert size < 1000000, f"File too large: {size} >= 1000000"
print(f"Headroom under 1,000,000: {1000000 - size:,} bytes")

compile(final_src, out_path, 'exec')
print("Syntax check: PASSED!")
