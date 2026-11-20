import re

with open('submission.py', 'r', encoding='utf-8') as f1, open('d:/OpenKernelChallenge/inference_research/keell0renz_muse/kernels/muse_moe_w4a16.py', 'r', encoding='utf-8') as f2:
    t1 = f1.read()
    t2 = f2.read()

d1 = dict(re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', t1))
d2 = dict(re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', t2))

print(f"{'Macro':<20} | {'submission.py':<15} | {'keell0renz':<15}")
print('-' * 55)
for k in sorted(set(d1.keys()) | set(d2.keys())):
    if any(x in k for x in ['WGS', 'MOE', 'CEPI', 'B4', 'MF', 'SC6']):
        v1 = d1.get(k, 'N/A').strip()
        v2 = d2.get(k, 'N/A').strip()
        if v1 != v2:
            print(f"{k:<20} | {v1:<15} | {v2:<15}")
