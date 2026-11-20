with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "OKC_MULDEC" in l:
        print(f"Line {i+1}:")
        for j in range(max(0, i-5), min(len(lines), i+25)):
            print(f"  {lines[j]}", end="")
        print("="*40)
