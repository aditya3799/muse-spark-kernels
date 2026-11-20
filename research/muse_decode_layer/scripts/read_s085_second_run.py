import json

with open('profiles/submission_s085_q1_occ3.9434c003de174c1f.json') as f:
    d = json.load(f)

for b in d.get('benchmarks', []):
    c = b['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t = b.get('median_ns') / 1000.0
    sp = b.get('speedup')
    print(f"{name:<35} {t:<10.2f} {sp:<10.4f}")
