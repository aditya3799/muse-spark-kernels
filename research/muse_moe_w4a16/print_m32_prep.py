with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("def _prepare_v167_m32_layout")
end = text.find("def prepare(")
print(text[pos:end])
