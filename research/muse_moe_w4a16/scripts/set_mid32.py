with open("submission.py") as f:
    text = f.read()

text = text.replace("MID_MAX_M = 16", "MID_MAX_M = 32", 1)
with open("submission.py", "w") as f:
    f.write(text)

print("Set MID_MAX_M = 32")
