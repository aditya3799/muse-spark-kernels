with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
matches = re.findall(r".{0,40}cublasLtMatmul.{0,40}", text)
print(f"cublasLtMatmul occurrences: {len(matches)}")
for m in matches[:10]:
    print(" ", m.strip())

matches2 = re.findall(r".{0,40}_nvjet.{0,40}", text)
print(f"_nvjet occurrences: {len(matches2)}")
for m in matches2[:10]:
    print(" ", m.strip())
