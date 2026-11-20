with open("submission.py") as f:
    text = f.read()

import re
matches = re.findall(r".{0,50}\.okc_run\(.{0,60}", text)
print(f"okc_run calls: {len(matches)}")
for m in matches:
    print(" ", m.strip())
