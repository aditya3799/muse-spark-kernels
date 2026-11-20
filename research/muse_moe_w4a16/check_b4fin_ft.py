with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]

lines = src.split('\\n')
for i, l in enumerate(lines):
    if 'B4FIN' in l or 'B4_' in l or 'sspart' in l:
        for j in range(max(0, i-5), min(len(lines), i+15)):
            if 'FT_T' in lines[j]:
                print(f"L{j}: {lines[j]}")
