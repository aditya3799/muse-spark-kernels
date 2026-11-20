import json
import math

with open('profiles/submission_s085_q1_occ3.9434c003de174c1f.json') as f:
    d = json.load(f)

loads = {}
for r in d.get('benchmarks', []):
    c = r['case']
    load = c['load']
    layer = c['layer']
    weight = c['weight']
    med_ns = r['median_ns']
    base_ns = r['baseline_ns']
    loads.setdefault(load, []).append((layer, weight, med_ns, base_ns))

load_speedups = {}
for load, items in loads.items():
    total_time = sum(w * t for _, w, t, _ in items)
    total_base = sum(w * b for _, w, _, b in items)
    sp = total_base / total_time
    load_speedups[load] = sp

vals = list(load_speedups.values())
geomean_sp = math.exp(sum(math.log(v) for v in vals) / len(vals))
print(f"S085 run 2 calculated speedup: {geomean_sp:.6f}")
