import re

with open("submission.py") as f:
    text = f.read()

m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
src = m.group(1).encode('utf-8').decode('unicode_escape')
lines = src.split('\n')
print("Lines 180-195:")
for idx in range(180, 195):
    print(f"L{idx:4d}: {lines[idx-1]}")
print("\nLines 2565-2580:")
for idx in range(2565, 2580):
    print(f"L{idx:4d}: {lines[idx-1]}")
