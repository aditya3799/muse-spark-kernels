with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("def prepare(")
end = text.find("def kernel_fn(")
prep = text[pos:end]
for i, l in enumerate(prep.splitlines()):
    if "KPERM" in l or "wAs" in l or "w[12]" in l or "wBs" in l or "w12" in l or "w13" in l:
        print(f"Line {i}: {l}")
