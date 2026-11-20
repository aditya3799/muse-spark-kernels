with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("ALT_SRCS['v167_wsi']")
end = text.find("ALT_SRCS['v167_helper0']")
lines = text[pos:end].replace("\\n", "\n").splitlines()

for i in range(1500, 1530):
    if i < len(lines):
        print(f"Line {i}: {lines[i]}")
