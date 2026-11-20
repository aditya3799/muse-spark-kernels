import difflib
import re

with open('submission_s081_keell0renz_890809c.py') as f1:
    s081 = f1.read()

with open('/mnt/d/OpenKernelChallenge/muse_attention_decode/submission.py') as f2:
    att = f2.read()

src1 = re.search(r'ATT7_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', s081).group(1)
src2 = re.search(r'CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', att).group(1)

l1 = src1.splitlines(keepends=True)
l2 = src2.splitlines(keepends=True)

diff = list(difflib.unified_diff(l1, l2, n=2))
print("Diff lines count:", len(diff))
for line in diff[:100]:
    print(line, end='')
