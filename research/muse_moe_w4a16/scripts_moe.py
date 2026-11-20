with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v172_hybrid_small.py') as f:
    text = f.read()

import re
moe_fn = re.findall(r'def _moe\(.*?\n\n', text, re.DOTALL)
for fn in moe_fn:
    print(fn)
