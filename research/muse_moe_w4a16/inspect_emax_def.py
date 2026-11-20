with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_wsi'] = '")
end = text.find("'\n", idx)
src = text[idx:end]

lines = [l for l in src.split('\\n') if 'EMAX' in l or 'MMAX' in l or 'JMAX' in l]
for l in lines[:15]:
    print(l)
