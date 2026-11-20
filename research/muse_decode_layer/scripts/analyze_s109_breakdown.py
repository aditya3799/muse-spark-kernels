import re
from collections import defaultdict

lines = open('profiles/submission_s109_cont1.2a4609207ae6410e.txt', encoding='utf-8', errors='ignore').readlines()

categories = defaultdict(float)

for line in lines:
    line = line.strip()
    if 'total' in line or 'GPU busy' in line or 'calls' in line or 'profile ' in line or not line:
        continue
    # find us: e.g. "28.8 µs" or "28.8 us"
    m = re.search(r'([\d\.]+)\s*(?:µs|us)\s+([\d\.]+)%\s+(\d+)', line)
    if m:
        us = float(m.group(1))
        name = line[:m.start()].strip()
        # classify category
        if 'moe_fused' in name:
            cat = 'MoE (okc::moe_fused)'
        elif 'attn_kernel' in name:
            cat = 'Attention (attn_kernel)'
        elif 'sk2_kernel' in name:
            cat = 'Skinny GEMM (sk2)'
        elif 'cgemv_kernel' in name:
            cat = 'CGEMV GEMV'
        elif 'glue_' in name or 'graft_' in name or 'route' in name or 'finish' in name:
            cat = 'Boundary / Routing Glue'
        elif 'rq<' in name or 'rqrun' in name:
            cat = 'RS128/RS32 RQ GEMM'
        else:
            cat = 'Other: ' + name[:30]
        categories[cat] += us

print(f"{'Category':<35} {'Total GPU (us)':<15}")
print("="*50)
total_all = sum(categories.values())
for cat, us in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat:<35} {us:<15.1f} ({us/total_all*100:.1f}%)")
print("="*50)
print(f"{'TOTAL':<35} {total_all:<15.1f}")
