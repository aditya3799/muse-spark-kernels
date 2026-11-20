with open('submission_s102_rq_cgemv.py') as f:
    text = f.read()

idx = text.find('RQBAKE =')
print(text[idx-200:idx+600])
