with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_wsi'] = '")
end = text.find("'\n", idx)
wsi = text[idx:end]
lines = [l for l in wsi.split('\\n') if 'OKC_WSNR' in l or 'OKC_WSBL' in l or 'OKC_WSB' in l]
for l in lines[:20]:
    print(l)
