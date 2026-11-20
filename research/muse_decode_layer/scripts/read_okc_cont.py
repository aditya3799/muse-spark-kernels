with open('submission.py') as f:
    text = f.read()

idx = text.find('#define OKC_CONT')
print(text[idx:idx+1500])
