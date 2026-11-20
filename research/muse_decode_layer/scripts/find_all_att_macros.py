with open('/mnt/d/OpenKernelChallenge/muse_attention_decode/submission.py') as f:
    att_top = f.read()

with open('submission.py') as f:
    dec = f.read()

start = dec.find('ATT7_CUDA_SRC = r"""')
end = dec.find('"""\nATT7_CPP_SRC', start)
dec_att = dec[start:end]

import re
m1_all = dict(re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', att_top))
m2_all = dict(re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', dec_att))

for k in sorted(set(m1_all.keys()) | set(m2_all.keys())):
    v1 = m1_all.get(k, 'MISSING').strip()
    v2 = m2_all.get(k, 'MISSING').strip()
    if v1 != v2:
        print(f"{k:<20} Standalone={v1:<15} Decoder={v2:<15}")
