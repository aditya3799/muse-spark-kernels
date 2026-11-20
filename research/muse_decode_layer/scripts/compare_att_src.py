import os
import re
import hashlib

with open('submission.py', 'r', encoding='utf-8') as f1:
    s = f1.read()

att_path = os.path.join('..', 'muse_attention_decode', 'submission.py')
with open(att_path, 'r', encoding='utf-8') as f2:
    att = f2.read()

m1 = re.search(r'ATT7_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', s)
m2 = re.search(r'CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', att)

if m1 and m2:
    s1, s2 = m1.group(1), m2.group(1)
    print(f"ATT7 len: {len(s1)}, att_decode len: {len(s2)}")
    print(f"Identical? {s1 == s2}")
    if s1 != s2:
        # find diff
        import difflib
        diff = list(difflib.unified_diff(s1.splitlines()[:50], s2.splitlines()[:50], lineterm=''))
        print("Diff in first 50 lines:")
        for line in diff[:30]:
            print(line)
else:
    print("Could not find sources", bool(m1), bool(m2))
