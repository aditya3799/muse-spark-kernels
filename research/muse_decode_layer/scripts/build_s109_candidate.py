"""Build submission S109:
1. Base: S108 (submission.py, which officially achieved 2.2684x / 9971 us, Rank 166)
2. Enable OKC_CONT 1: continuous stage pipeline across segments in flat attention (no pipeline drain per segment)
3. Module name: muse_att_v298w1pf_s109_{arch}
4. Save with LF line endings, strictly < 1,000,000 bytes.
"""

import os
import re

with open('submission.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Enable OKC_CONT 1
cont_old = '#define OKC_CONT 0      // flat producer mode: one continuous stage pipeline across segments'
cont_new = '#define OKC_CONT 1      // flat producer mode: one continuous stage pipeline across segments'
assert cont_old in src, "OKC_CONT not found"
src = src.replace(cont_old, cont_new, 1)

# 2. Update module name for cache freshness
mod_old = 'load_inline(name=f"muse_att_v298w1pf_s108_{arch}"'
mod_new = 'load_inline(name=f"muse_att_v298w1pf_s109_{arch}"'
assert mod_old in src, "mod name not found"
src = src.replace(mod_old, mod_new, 1)

# 3. Update prepare linking if present
src = src.replace("_s108_prior_prepare = prepare", "_s109_prior_prepare = prepare", 1)
src = src.replace("_s073_prior_prepare = _s108_prior_prepare", "_s073_prior_prepare = _s109_prior_prepare", 1)

out_path = 'submission_s109_cont1.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(src)

size = os.path.getsize(out_path)
print(f"Built {out_path}: {size:,} bytes")
assert size < 1000000, f"File too large: {size} >= 1000000"
print(f"Headroom under 1,000,000: {1000000 - size:,} bytes")

compile(src, out_path, 'exec')
print("Syntax check: PASSED!")
