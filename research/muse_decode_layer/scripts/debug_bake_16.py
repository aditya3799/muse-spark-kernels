with open('submission_s112_rq_qkv_l1.py') as f:
    text = f.read()

idx = text.find('def _rqbake(')
print(text[idx:idx+2500])
