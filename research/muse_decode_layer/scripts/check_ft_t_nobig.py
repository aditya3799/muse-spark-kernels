with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
print("OKC_FT_T in submission.py:")
for m in re.finditer(r".{0,40}OKC_FT_T.{0,40}", text):
    print(" ", m.group(0).strip())

print("\nOKC_NO_BIG in submission.py:")
for m in re.finditer(r".{0,40}OKC_NO_BIG.{0,40}", text):
    print(" ", m.group(0).strip())
