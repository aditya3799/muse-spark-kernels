import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
print("Lines 1715-1740:")
for idx in range(1715, 1740):
    print(f"L{idx:4d}: {lines[idx-1]}")
print("\nLines 2280-2305:")
for idx in range(2280, 2305):
    print(f"L{idx:4d}: {lines[idx-1]}")
