import re

with open("submission.py") as f:
    orig = f.read()

idx = orig.find("ALT_SRCS['v167_main'] = '")
v_idx = orig.find("V167_MAIN_DEFINES", idx)
v167_main = orig[idx:v_idx]

lines = v167_main.split('\\n')
for idx, l in enumerate(lines, 1):
    if 'WGS_CA' in l or 'wgs_unitc' in l or 'wtot' in l:
        print(f"L{idx:4d}: {l}")
