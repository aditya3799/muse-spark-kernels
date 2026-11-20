with open('submission.py') as f:
    text = f.read()

idx = text.find('def bfp_kernel_fn')
print(text[idx:idx+1500])
