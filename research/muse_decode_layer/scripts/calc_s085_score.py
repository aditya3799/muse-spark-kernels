import json
import math

def calc_score(records):
    loads = {}
    for r in records:
        c = r['case']
        load = c['load']
        layer = c['layer']
        weight = c['weight']  # 46 or 16
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
    return geomean_sp

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s085_q1_occ3.6ea69b33f4d747a3.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

print("S081 sp:", calc_score(d1.get('benchmarks', [])))
print("S085 sp:", calc_score(d2.get('benchmarks', [])))
