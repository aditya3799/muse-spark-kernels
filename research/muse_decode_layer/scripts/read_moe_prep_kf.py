with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i in range(4380, min(4450, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
