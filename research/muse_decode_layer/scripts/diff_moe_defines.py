with open('submission_s081_keell0renz_890809c.py') as f:
    text1 = f.read()

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v178_hybrid_m32_retry.py') as f:
    text2 = f.read()

import re
def extract_defines(t):
    defs = {}
    for line in t.splitlines():
        m = re.match(r'^\s*#define\s+(OKC_\w+)\s+(.+)', line)
        if m:
            defs[m.group(1)] = m.group(2).split('//')[0].strip()
    return defs

d1 = extract_defines(text1)
d2 = extract_defines(text2)

print("Differences in OKC_ defines (S081 vs v178):")
for k in sorted(set(d1.keys()) | set(d2.keys())):
    v1 = str(d1.get(k))
    v2 = str(d2.get(k))
    if v1 != v2:
        print(f"  {k:<20}: S081 = {v1:<15} v178 = {v2}")
