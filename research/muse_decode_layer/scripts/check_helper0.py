with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('_MA_ALT_SRCS["moe_helper0"]')
print('Found at', idx)
if idx != -1:
    end = text.find('_MA_ALT_SRCS["moe_helper0df"]', idx)
    sub = text[idx:end]
    print('Contains OKC_WGSU:', 'OKC_WGSU' in sub)
    if 'OKC_WGSU' in sub:
        for line in sub.splitlines():
            if 'OKC_WGSU' in line:
                print(line)
