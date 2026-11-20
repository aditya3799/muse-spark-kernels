with open('submission_s081_keell0renz_890809c.py') as f1:
    s081 = f1.read()

with open('/mnt/d/OpenKernelChallenge/muse_attention_decode/submission.py') as f2:
    att = f2.read()

import re
m1 = re.search(r'ATT7_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', s081)
m2 = re.search(r'CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', att)

lines1 = m1.group(1).splitlines()[:60]
lines2 = m2.group(1).splitlines()[:60]

print("--- ATT7_CUDA_SRC (in decode layer) ---")
for l in lines1[:25]:
    print(l)

print("\n--- CUDA_SRC (in attention decode standalone) ---")
for l in lines2[:25]:
    print(l)
