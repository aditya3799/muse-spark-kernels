import re

with open('../muse_moe_w4a16/submission.py', 'r', encoding='utf-8') as f:
    s = f.read()

# find how _ext is called or how M >= 128 is handled
m = re.findall(r'(def _prepare_layout[\s\S]*?def kernel_fn[\s\S]*?\n\n)', s)
if m:
    print("Found kernel_fn and prepare block in muse_moe_w4a16:")
    print(m[0][:1500])
else:
    print("Searching for MMAX or 128...")
    for line in s.splitlines():
        if '128' in line and ('def' in line or 'ext' in line or 'DEFINES' in line or 'MMAX' in line):
            print(" ", line)
