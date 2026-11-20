import json
from collections import defaultdict

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

comp_time = defaultdict(float)
total_time = 0.0

for p in d.get('profiles', []):
    for line in p['report'].splitlines():
        line = line.strip()
        if not line or line.startswith('kernel') or line.startswith('total') or line.startswith('GPU busy') or line.startswith('an overlapping'):
            continue
        parts = line.split()
        try:
            # find the time in us
            us_idx = [i for i, part in enumerate(parts) if part == 'µs']
            if us_idx:
                t = float(parts[us_idx[0]-1])
                name = " ".join(parts[:us_idx[0]-1])
                if 'moe_fused' in name:
                    comp = 'MoE'
                elif 'attn_kernel' in name:
                    comp = 'Attention'
                elif 'sk2_kernel' in name:
                    comp = 'sk2 GEMM'
                elif 'rq<' in name:
                    comp = 'rq GEMM'
                elif 'nvjet' in name:
                    comp = 'nvjet GEMM'
                elif 'boundary' in name or 'graft' in name or 'route' in name:
                    comp = 'Boundary / Routing'
                elif 'cgemv' in name:
                    comp = 'cgemv'
                else:
                    comp = 'Other (' + name[:20] + ')'
                comp_time[comp] += t
                total_time += t
        except Exception:
            pass

print(f"{'Component':<25} {'Total GPU us':<15} {'Percentage':<10}")
print("="*50)
for comp, t in sorted(comp_time.items(), key=lambda x: -x[1]):
    print(f"{comp:<25} {t:<15.1f} {t/total_time*100:<10.1f}%")
print(f"{'TOTAL':<25} {total_time:<15.1f}")
