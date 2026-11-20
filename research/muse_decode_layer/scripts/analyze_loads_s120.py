import json
from collections import defaultdict

with open('profiles/submission_s120_wsi_fpart.4541145832f0451f.json') as f:
    d = json.load(f)

profiles = d.get('profiles', [])
if isinstance(profiles, dict):
    profiles = list(profiles.values())

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
kernel_calls = defaultdict(int)

for p_val in profiles:
    case = p_val.get('case', {})
    load = case.get('load')
    layer = case.get('layer')
    w = weights.get(load, {}).get(layer, 1)
    
    for k in p_val.get('kernels', []):
        name = k.get('name', '')
        time_us = k.get('time_ns', 0) / 1000.0
        calls = k.get('calls', 1)
        
        # Categorize
        cat = 'Other'
        if 'moe_fused' in name:
            cat = 'MoE (okc::moe_fused)'
        elif 'attn_kernel' in name:
            cat = 'Attention (attn_kernel)'
        elif 'sk2_kernel' in name:
            cat = 'Skinny GEMM (sk2_kernel)'
        elif 'rq<' in name or 'rq_' in name:
            cat = 'RQ GEMM (rq<M>)'
        elif 'nvjet' in name:
            cat = 'cuBLAS / nvjet'
        elif 'cgemv' in name:
            cat = 'CGEMV (12-bit)'
        elif 'boundary' in name or 'cluster' in name or 'route' in name:
            cat = 'Boundary & Route'
        
        kernel_weighted_time[cat] += time_us * w
        kernel_raw_time[cat] += time_us
        kernel_calls[cat] += calls

total_weighted = sum(kernel_weighted_time.values())
total_raw = sum(kernel_raw_time.values())

print(f"{'Category':<30} | {'Weighted (us)':<14} | {'Share %':<8} | {'Raw (us)':<10}")
print('-' * 70)
for cat, w_time in sorted(kernel_weighted_time.items(), key=lambda x: x[1], reverse=True):
    share = (w_time / total_weighted) * 100
    raw = kernel_raw_time[cat]
    print(f"{cat:<30} | {w_time:<14.1f} | {share:<7.1f}% | {raw:<10.1f}")
print('-' * 70)
print(f"{'TOTAL':<30} | {total_weighted:<14.1f} | {'100.0%':<8} | {total_raw:<10.1f}")
