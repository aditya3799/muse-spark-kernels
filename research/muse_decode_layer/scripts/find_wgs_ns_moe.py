with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
matches = re.findall(r".{0,40}WGS_N[SX].{0,40}", text)
for m in set(matches):
    print(m.strip())
