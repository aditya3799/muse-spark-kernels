with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "attn_kernel(" in l:
        print(f"Line {i+1}:")
        for j in range(max(0, i-6), min(len(lines), i+10)):
            print(f"  {lines[j]}", end="")
        break
