with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("ALT_SRCS['v167_wsi'] = '")
if idx != -1:
    end = text.find("'\n", idx)
    wsi = text[idx:end]
    print("Found v167_wsi, length:", len(wsi))
    lines = [l for l in wsi.split('\\n') if 'OKC_FAST1' in l or 'OKC_WSBLK' in l or 'OKC_WSBL' in l]
    for l in lines[:20]:
        print(l)
else:
    print("v167_wsi not found")
