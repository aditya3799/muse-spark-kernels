with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "def _plan7" in l:
        print(f"Line {i+1}:")
        for j in range(i, min(i+60, len(lines))):
            print(f"  {lines[j]}", end="")
        print("="*40)
