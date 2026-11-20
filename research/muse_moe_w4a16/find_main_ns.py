with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

for i, l in enumerate(text.splitlines()):
    if "MAIN_NS" in l:
        print(f"Line {i}: {l}")
