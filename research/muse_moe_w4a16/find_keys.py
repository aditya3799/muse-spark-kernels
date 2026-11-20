import re

with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

keys = re.findall(r"ALT_SRCS\['([^']+)'\]\s*=", text)
print("ALT_SRCS keys:", keys)

for k in keys:
    print(f"Key: {k}")
