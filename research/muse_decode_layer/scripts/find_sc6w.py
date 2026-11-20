with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "_MA_SC6W" in l:
        print(f"Line {i+1}: {l.strip()}")
