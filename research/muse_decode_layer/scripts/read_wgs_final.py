with open('submission_s109_cont1.py') as f:
    text = f.read()

idx = text.find('// final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]')
print(text[idx:idx+2000])
