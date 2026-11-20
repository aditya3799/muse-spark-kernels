with open("submission_s122_att_qdir_pk2.py") as f:
    t122 = f.read()

with open("submission.py") as f:
    t125 = f.read()

import re

m122 = re.search(r"def _moe\(.*?\n\n", t122, re.DOTALL)
m125 = re.search(r"def _moe\(.*?\n\n", t125, re.DOTALL)

print("=== S122 _moe ===")
if m122: print(m122.group(0))

print("=== S125 _moe ===")
if m125: print(m125.group(0))
