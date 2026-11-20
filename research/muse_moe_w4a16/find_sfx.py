with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("def _prepare_layout")
end = text.find("def _prepare_v167_m32_layout")
for i, l in enumerate(text[pos:end].splitlines()):
    if "sfx" in l:
        print(f"Line {i}: {l}")
