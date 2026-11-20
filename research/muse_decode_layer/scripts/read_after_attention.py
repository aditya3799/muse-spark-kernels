with open('submission.py') as f:
    text = f.read()

idx = text.find('def _after_attention')
print(text[idx:idx+2500])
