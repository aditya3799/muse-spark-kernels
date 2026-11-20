with open('submission.py') as f:
    text = f.read()

idx = text.find('def kernel_fn(data, state):')
print(text[idx:idx+800])
