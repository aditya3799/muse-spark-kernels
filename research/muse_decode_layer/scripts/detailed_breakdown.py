import re

with open('profiles/submission_s114_rq_qkv_fixed2.843243d902ab4b5f.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sections = re.split(r'profile \d+:', text)[1:]

categories = {'moe': 0.0, 'attn': 0.0, 'gemm_sk2': 0.0, 'gemm_rq': 0.0, 'gemm_cgemv': 0.0, 'boundary_route': 0.0, 'other': 0.0}

shape_weights = [46, 16, 46, 16, 46, 16, 46, 16, 46, 16, 46, 16]

weighted_cat = {k: 0.0 for k in categories}

for i, sec in enumerate(sections[:12]):
    w = shape_weights[i]
    for line in sec.strip().splitlines()[2:]:
        if line.startswith('total'):
            break
        # Match time in us (e.g. 58.4 µs or 58.4 us)
        m = re.search(r'(\d+\.\d+)\s*(?:µs|us)', line)
        if not m:
            continue
        t_us = float(m.group(1))
        name = line[:m.start()].strip()
        
        if 'moe_fused' in name:
            c = 'moe'
        elif 'attn_kernel' in name:
            c = 'attn'
        elif 'sk2_kernel' in name:
            c = 'gemm_sk2'
        elif 'rq<' in name:
            c = 'gemm_rq'
        elif 'cgemv' in name:
            c = 'gemm_cgemv'
        elif any(x in name for x in ['route', 'boundary', 'cluster', 'sums', 'apply', 'prep']):
            c = 'boundary_route'
        else:
            c = 'other'
        
        categories[c] += t_us
        weighted_cat[c] += w * t_us

print(f"{'Category':<18} | {'Raw Sum (us)':<12} | {'Raw %':<8} | {'Weighted (us)':<14} | {'Weighted %':<10}")
print('-' * 70)
total_raw = sum(categories.values())
total_w = sum(weighted_cat.values())
for k in sorted(categories.keys(), key=lambda x: weighted_cat[x], reverse=True):
    raw = categories[k]
    raw_pct = (raw / total_raw) * 100.0 if total_raw > 0 else 0
    wt = weighted_cat[k]
    wt_pct = (wt / total_w) * 100.0 if total_w > 0 else 0
    print(f"{k:<18} | {raw:<12.1f} | {raw_pct:<7.1f}% | {wt:<14.1f} | {wt_pct:<9.1f}%")
