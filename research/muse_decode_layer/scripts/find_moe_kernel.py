with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
m = re.findall(r'__global__.*?void.*?moe_fused.*?\(', text)
for f in m:
    print(f)
