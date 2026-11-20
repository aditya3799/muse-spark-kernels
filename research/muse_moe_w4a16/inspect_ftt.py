with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]
lines = src.split('\\n')
for i, l in enumerate(lines):
    if 'FT_T' in l or 'wgs_range' in l or 'wgs_plan' in l:
        print(f"L{i}: {l}")
