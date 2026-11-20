with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("ALT_SRCS['v167_helper0']")
end = text.find("ALT_SRCS['v167_main']")
lines = text[pos:end].replace("\\n", "\n").splitlines()

for i in range(120, 200):
    if i < len(lines):
        print(f"{i}: {lines[i]}")
