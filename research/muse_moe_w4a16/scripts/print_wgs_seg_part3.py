import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
print("Lines 2200-2250:")
for idx in range(2200, 2250):
    print(f"L{idx:4d}: {lines[idx-1]}")
