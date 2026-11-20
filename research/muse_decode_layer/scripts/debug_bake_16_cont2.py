with open('submission_s112_rq_qkv_l1.py') as f:
    text = f.read()

idx = text.find('INVALID %s" % (layer, m, n')
print(text[idx:idx+2000])
