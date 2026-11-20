with open('submission.py') as f:
    text = f.read()

idx = text.find('def dense(')
print(text[idx:idx+1200])
