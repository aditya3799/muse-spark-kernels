with open('submission.py') as f:
    text = f.read()

idx = text.find('def _rq_gemm')
print(text[idx-200:idx+1500])
