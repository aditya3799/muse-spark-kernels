import json

with open('check_s123.json') as f:
    d = json.load(f)

pj = json.loads(d['profile']['json'])
cases = pj.get('cases', [])
print(f"Total profiled cases: {len(cases)}")
for i, c in enumerate(cases):
    print(f"Case {i}: {c.get('name')} | GPU busy: {c.get('gpu_busy_us'):.2f} us")
    for k in c.get('kernels', []):
        if any(w in k['name'] for w in ['moe', 'attn', 'sk2', 'cgemv', 'nvjet']):
            print(f"   {k['name'][:55]:55s} | {k['dur_us']:6.2f} us ({k['share_pct']:.1f}%)")
