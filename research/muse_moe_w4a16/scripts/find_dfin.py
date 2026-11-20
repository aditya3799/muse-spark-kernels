import re

with open("submission.py") as f:
    orig = f.read()

idx = orig.find("ALT_SRCS['v167_main'] = '")
v_idx = orig.find("V167_MAIN_DEFINES", idx)
v167_main = orig[idx:v_idx]

for l in v167_main.split('\\n'):
    if 'WGSDFIN' in l or 'DFIN' in l:
        print(l)
