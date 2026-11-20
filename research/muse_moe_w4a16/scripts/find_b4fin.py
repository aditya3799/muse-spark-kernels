import re

with open("submission.py") as f:
    text = f.read()

# find v167_main
m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", text, re.DOTALL)
if m:
    src = m.group(1).encode('utf-8').decode('unicode_escape')
    lines = src.split('\n')
    for idx, l in enumerate(lines, 1):
        if 'B4FIN' in l or 'sspart' in l:
            print(f"L{idx:4d}: {l}")
else:
    print("Not found")
