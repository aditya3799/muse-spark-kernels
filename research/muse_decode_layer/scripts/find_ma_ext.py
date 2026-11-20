with open("submission.py") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "ma_ext" in l:
        print(f"{i+1}: {l.strip()}")
