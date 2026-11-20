with open('submission_s109_cont1.py') as f:
    text = f.read()

idx = text.find('// final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]')
idx2 = text.find('#if OKC_WGS && OKC_B4FIN', idx)
print(text[idx2:idx2+4000])
