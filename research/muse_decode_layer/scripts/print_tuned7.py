with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "_TUNED7" in l:
        print(f"Line {i+1}: {l.strip()}")
        for j in range(i+1, min(i+35, len(lines))):
            print(f"  {lines[j].strip()}")
            if "}" in lines[j]:
                break
        break
