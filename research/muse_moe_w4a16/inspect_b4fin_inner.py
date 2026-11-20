with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]
lines = src.split('\\n')
for i, l in enumerate(lines):
    if 'constexpr int SEG = 1024' in l:
        print('\\n'.join(lines[i:i+60]))
        break
