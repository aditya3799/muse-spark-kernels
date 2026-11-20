with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
matches = re.findall(r"#if\s+OKC_WGS.*?\n", text)
print("OKC_WGS ifdefs:", len(matches))
for m in set(matches):
    print(" ", m.strip())
