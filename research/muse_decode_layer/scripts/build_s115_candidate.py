with open('submission_s114_rq_qkv_fixed2.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update module name for attention extension
text = text.replace('muse_att_v298w1pf_s114_{arch}', 'muse_att_v298w1pf_s115_{arch}')

# 2. Enable OKC_CONT = 1
old_cont = '#define OKC_CONT 0      // flat producer mode: one continuous stage pipeline across segments (no drain per segment)'
assert old_cont in text, "old_cont not found"
new_cont = '#define OKC_CONT 1      // flat producer mode: one continuous stage pipeline across segments (no drain per segment)'
text = text.replace(old_cont, new_cont, 1)

out_file = 'submission_s115_cont1_rq_qkv.py'
with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

import os
size = os.path.getsize(out_file)
print(f"Successfully generated {out_file}, size = {size} bytes")
assert size < 1000000, f"Size {size} exceeds 1,000,000 bytes!"
