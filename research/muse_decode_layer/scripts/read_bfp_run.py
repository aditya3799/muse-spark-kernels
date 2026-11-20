with open('submission.py') as f:
    text = f.read()

idx = text.find('def bfp__run(')
print(text[idx:idx+1500])
