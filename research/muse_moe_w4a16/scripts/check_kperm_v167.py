import re

with open("submission.py") as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

lines = v167_main.split('\\n')
for idx in range(2160, 2215):
    print(f"L{idx:4d}: {lines[idx-1]}")
