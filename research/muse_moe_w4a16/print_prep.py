with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
pos = -1
for i, l in enumerate(lines):
    if l.startswith("def prepare("):
        pos = i
        break

for i in range(pos, min(len(lines), pos + 100)):
    print(f"Line {i}: {lines[i]}")
