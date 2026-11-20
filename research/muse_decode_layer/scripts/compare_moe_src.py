with open('submission.py', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

import re
m1 = re.search(r'MOE_CUDA_SRC\s*=\s*r?("""[\s\S]*?""")', text)
s1 = m1.group(1)[3:-3] if m1 else ''

# moe_helper0 is on line 12583
m2 = re.search(r'_MA_ALT_SRCS\["moe_helper0"\]\s*=\s*(\'[\s\S]*?\')\n\s*MOE_MID_DEFINES', text)
s2 = eval(m2.group(1)) if m2 else ''

print(f"MOE_CUDA_SRC len: {len(s1):,}, moe_helper0 len: {len(s2):,}")
import difflib
diff = list(difflib.unified_diff(s1.splitlines(keepends=True), s2.splitlines(keepends=True), n=1))
print("Diff lines:", len(diff))
for l in diff[:40]:
    print(l, end='')
