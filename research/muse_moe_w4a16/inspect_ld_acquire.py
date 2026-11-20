with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_main'] = '")
end = text.find("'\n", idx)
src = text[idx:end]

lines = [l for l in src.split('\\n') if 'ld_acquire' in l]
for l in lines[:10]:
    print(l)
