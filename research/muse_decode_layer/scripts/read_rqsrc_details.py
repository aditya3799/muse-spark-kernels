with open('submission.py') as f:
    text = f.read()

idx = text.find('def _rqsrc(vs):')
print(text[idx-300:idx+1500])
