with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]

lines = src.split('\\n')
for i, l in enumerate(lines):
    if 'static_assert(NT == 512 && NW == 16, "B4FIN' in l:
        print('\\n'.join(lines[i-5:i+55]))
        break
