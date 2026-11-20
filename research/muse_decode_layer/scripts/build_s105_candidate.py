"""Build submission S105:
1. Base: submission_s081_keell0renz_890809c.py
2. Retain full _mf_ab with MFAB_LAYER_REPS = 24, MFAB_LAYER_SEEDS = 4 (full warmup)
3. Keep RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)} (fast PDL sk2 path for m=16 and m=32)
4. Tune MoE stream-K granularity from v178:
   OKC_WGSU: 160 -> 80
   OKC_WGSUB: 140 -> 70
5. Append validated exact-C12 GEMV for T=1 from S073/S090.
6. Save with LF line endings, strictly < 1,000,000 bytes.
"""

import os
import re

# 1. Read base S081
with open('submission_s081_keell0renz_890809c.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 2. Update MoE stream-K granularity
wgsu_old = '#define OKC_WGSU 160   // cepi: per-unit cost, phase A (zhebrak SK_CEPI)\n#define OKC_WGSUB 140  // cepi: per-unit cost, phase B'
wgsu_new = '#define OKC_WGSU 80    // cepi: per-unit cost, phase A (zhebrak SK_CEPI)\n#define OKC_WGSUB 70   // cepi: per-unit cost, phase B'
assert wgsu_old in src, "OKC_WGSU definition not found"
src = src.replace(wgsu_old, wgsu_new, 1)

# Also check if OKC_WGSU is referenced anywhere else in the file:
# In wgs_prefix and wgs_ranges:
# acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
# accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
# ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
# ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
# Those use the defined OKC_WGSU macro directly!

# 3. Read CGEMV appendix from S090
with open('submission_s090_s081_cgemv_m1.py', 'r', encoding='utf-8') as f:
    s090 = f.read()

marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
appendix_pos = s090.rfind("# ---------------------------------------------------------------------------", 0, s090.index(marker))
cgemv_appendix = s090[appendix_pos:].rstrip()

# In CGEMV appendix, ensure prior prepare chain links to s105
cgemv_appendix = cgemv_appendix.replace("_s073_prior_prepare = prepare", "_s105_prior_prepare = prepare\n_s073_prior_prepare = _s105_prior_prepare", 1)

# 4. Combine
final_src = src.rstrip() + "\n\n" + cgemv_appendix + "\n"

# Verify size and syntax
out_path = 'submission_s105_cgemv_pdl.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(final_src)

size = os.path.getsize(out_path)
print(f"Built {out_path}: {size:,} bytes")
assert size < 1000000, f"File too large: {size} >= 1000000"
print(f"Headroom under 1,000,000: {1000000 - size:,} bytes")

# Python syntax check
compile(final_src, out_path, 'exec')
print("Syntax check: PASSED!")
