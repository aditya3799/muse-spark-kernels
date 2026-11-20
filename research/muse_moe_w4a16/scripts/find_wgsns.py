import re

with open("submission.py") as f:
    text = f.read()

idx_main = text.find("ALT_SRCS['v167_main'] = '")
idx_defines = text.find("V167_MAIN_DEFINES", idx_main)
v167_main = text[idx_main:idx_defines]

for l in v167_main.split('\\n'):
    if 'WGSNS' in l or 'WGS_NS' in l:
        print(l)
