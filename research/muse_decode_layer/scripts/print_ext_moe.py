with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "def _ext" in l:
        for j in range(i, min(i+70, len(lines))):
            print(f"{j+1}: {lines[j]}", end="")
        break
