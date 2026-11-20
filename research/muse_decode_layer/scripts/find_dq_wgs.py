with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
for m in re.finditer(r"// =+ WGS.*?void dq_wgs\(", text, re.DOTALL):
    print(m.group(0)[:500])

for m in re.finditer(r"void dq_wgs\(.*?\n\}", text, re.DOTALL):
    print(m.group(0))
