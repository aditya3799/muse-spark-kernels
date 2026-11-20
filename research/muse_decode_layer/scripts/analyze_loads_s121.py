import json
import re
from collections import defaultdict

with open('profiles/submission.6cafa69edc014362.json', encoding='utf-8') as f:
    d = json.load(f)

weights = {
    'decode 1': {'sliding': 46, 'full': 16},
    'decode 16': {'sliding': 46, 'full': 16},
    'decode 32': {'sliding': 46, 'full': 16},
    'verify 1': {'sliding': 46, 'full': 16},
    'verify 16': {'sliding': 46, 'full': 16},
    'verify 32': {'sliding': 46, 'full': 16},
}

kernel_weighted_time = defaultdict(float)
kernel_raw_time = defaultdict(float)

for p in d.get('profiles', []):
    case = p.get('case', {})
    load = case.get('load')
    layer = case.get('layer')
    w = weights.get(load, {}).get(layer, 1)
    
    report = p.get('report', '')
    for line in report.splitlines():
        if line.startswith('kernel') or line.startswith('total') or line.startswith('GPU busy') or not line.strip() or 'an overlapping' in line:
            continue
        m = re.search(r'^(.*?)\s+([\d\.]+)\s*\S+s\s+([\d\.]+)%', line.strip())
        if m:
            name = m.group(1).strip()
            time_us = float(m.group(2))
            
            cat = 'Other'
            if 'moe_fused' in name:
                cat = 'MoE (okc::moe_fused)'
            elif 'attn_kernel' in name:
                cat = 'Attention (attn_kernel)'
            elif 'sk2_kernel' in name:
                cat = 'Skinny GEMM (sk2)'
            elif 'rq<' in name or 'rq_' in name:
                cat = 'RQ GEMM (rq)'
            elif 'nvjet' in name:
                cat = 'cuBLAS / nvjet'
            elif 'cgemv' in name:
                cat = 'CGEMV (12-bit)'
            elif 'boundary' in name or 'cluster' in name or 'route' in name:
                cat = 'Boundary & Route'
            
            kernel_weighted_time[cat] += time_us * w
            kernel_raw_time[cat] += time_us

total_weighted = sum(kernel_weighted_time.values())
total_raw = sum(kernel_raw_time.values())

print(f"{'Category':<28} | {'Weighted (us)':<14} | {'Share %':<8} | {'Raw (us)':<10}")
print('-' * 68)
for cat, w_time in sorted(kernel_weighted_time.items(), key=lambda x: x[1], reverse=True):
    share = (w_time / total_weighted) * 100
    raw = kernel_raw_time[cat]
    print(f"{cat:<28} | {w_time:<14.1f} | {share:<7.1f}% | {raw:<10.1f}")
print('-' * 68)
print(f"{'TOTAL':<28} | {total_weighted:<14.1f} | {'100.0%':<8} | {total_raw:<10.1f}")
