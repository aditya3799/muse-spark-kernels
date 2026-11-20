with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

idx = text.find('template <int M,')
if idx != -1:
    print(text[idx:idx+2500])
else:
    idx2 = text.find('__global__ void rq')
    print(text[idx2:idx2+2500])
