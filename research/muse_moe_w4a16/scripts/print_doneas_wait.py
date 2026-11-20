import re

with open("submission.py") as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

lines = v167_main.split('\\n')
for idx, l in enumerate(lines, 1):
    if 'while (ld_acquire(mf_da' in l:
        for k in range(max(1, idx - 15), min(len(lines), idx + 20)):
            print(f"L{k:4d}: {lines[k-1]}")
