import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
for idx, l in enumerate(lines, 1):
    if 'WGS_XROWS' in l:
        print(f"L{idx:4d}: {l}")
