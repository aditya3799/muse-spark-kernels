with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
matches = re.findall(r".{0,40}OKC_W0.{0,40}", text)
for m in set(matches):
    print(m.strip())
