import json
import math

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('check_s109.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']

loads = {}
for i, r in enumerate(b2):
    c = r['case']
    load = c['load']
    layer = c['layer']
    weight = c['weight']
    med_ns = r['median_ns']
    base_ns = b1[i]['baseline_ns']  # Use standard reference baseline
    loads.setdefault(load, []).append((layer, weight, med_ns, base_ns))

load_speedups = {}
for load, items in loads.items():
    total_time = sum(w * t for _, w, t, _ in items)
    total_base = sum(w * b for _, w, _, b in items)
    sp = total_base / total_time
    load_speedups[load] = sp

vals = list(load_speedups.values())
geomean_sp = math.exp(sum(math.log(v) for v in vals) / len(vals))
print(f"S109 speedup on normalized baseline: {geomean_sp:.6f}")
