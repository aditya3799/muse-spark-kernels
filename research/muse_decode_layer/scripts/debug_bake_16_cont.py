with open('submission_s112_rq_qkv_l1.py') as f:
    text = f.read()

idx = text.find('esc2, mx2 = _rqesc2(')
print(text[idx:idx+2000])
