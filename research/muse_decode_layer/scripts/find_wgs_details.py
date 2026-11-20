with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
matches = re.findall(r".{0,40}wgs_step.{0,40}", text)
print("wgs_step matches:", len(matches))
for m in matches[:10]:
    print(" ", m.strip())

matches2 = re.findall(r"template\s*<.*?>\s*__device__\s*void\s*wgs_.*?\n", text)
print("wgs functions:", len(matches2))
for m in matches2[:10]:
    print(" ", m.strip())
