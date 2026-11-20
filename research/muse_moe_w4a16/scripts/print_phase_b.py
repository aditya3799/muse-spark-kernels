import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
for idx in range(2030, min(2070, len(lines))):
    print(f"L{idx:4d}: {lines[idx-1]}")
