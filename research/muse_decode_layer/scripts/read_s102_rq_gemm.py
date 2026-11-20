with open('submission_s102_rq_cgemv.py') as f:
    text = f.read()

idx = text.find('def _rq_gemm')
if idx != -1:
    print(text[idx:idx+1500])
else:
    print("Not found _rq_gemm")
