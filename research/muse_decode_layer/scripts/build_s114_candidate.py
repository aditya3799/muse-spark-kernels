with open('submission_s113_rq_qkv_fixed.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update module name for attention extension to avoid cache collisions
text = text.replace('muse_att_v298w1pf_s113_{arch}', 'muse_att_v298w1pf_s114_{arch}')

# 2. Add t32 and t16 to the SECOND RQPROD (the active one)
old_second_rqprod = '''    RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",
              "d2s5hzwP", "s4s5hzw"]'''
assert old_second_rqprod in text, "old_second_rqprod not found"
new_second_rqprod = '''    RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",
              "d2s5hzwP", "s4s5hzw", "t32s6hzw", "t32s5hzw", "t16s5hzw", "t16s4hzw"]'''
text = text.replace(old_second_rqprod, new_second_rqprod, 1)

out_file = 'submission_s114_rq_qkv_fixed2.py'
with open(out_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

import os
size = os.path.getsize(out_file)
print(f"Successfully generated {out_file}, size = {size} bytes")
assert size < 1000000, f"Size {size} exceeds 1,000,000 bytes!"
