with open('submission.py') as f:
    text = f.read()

idx = text.find('RQV = [')
print(text[idx:idx+3500])
