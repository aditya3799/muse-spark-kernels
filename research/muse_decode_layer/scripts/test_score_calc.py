import json
import math

def calc_score(records):
    # Group by load
    loads = {}
    for r in records:
        if r.get('event') != 'benchmark':
            continue
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
        # print(f"  {load}: time={total_time/62000:.2f}us base={total_base/62000:.2f}us sp={sp:.4f}")
    
    # Geomean of load speedups
    vals = list(load_speedups.values())
    geomean_sp = math.exp(sum(math.log(v) for v in vals) / len(vals))
    # Geomean score_us: base_geomean / geomean_sp ?
    # Let's check how OKC computes score_us:
    return geomean_sp, load_speedups

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)
sp, ls = calc_score(d.get('benchmarks', []))
print(f"S081 calc geomean speedup: {sp:.6f}")
print("S081 recorded speedup:    2.2611966")

with open('check_s104.json') as f:
    d = json.load(f)
records = d['result']['records']
sp, ls = calc_score(records)
print(f"S104 calc geomean speedup: {sp:.6f}")
print(f"S104 recorded speedup:    {d['result']['speedup']:.6f}")
