with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i in range(3560, min(3650, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
