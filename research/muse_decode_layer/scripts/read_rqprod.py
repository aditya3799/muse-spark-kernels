with open('submission.py') as f:
    text = f.read()

idx = text.find('RQPROD = [')
print(text[idx-400:idx+800])
