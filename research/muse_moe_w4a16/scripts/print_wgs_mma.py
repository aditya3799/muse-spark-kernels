import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
print("Lines 1770-1815:")
for idx in range(1770, 1815):
    print(f"L{idx:4d}: {lines[idx-1]}")
