import json
import math

def get_loads(records):
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
    
    res = {}
    for load, items in loads.items():
        total_time = sum(w * t for _, w, t, _ in items)
        total_base = sum(w * b for _, w, _, b in items)
        sp = total_base / total_time
        res[load] = (total_time / 62000.0, total_base / 62000.0, sp)
    return res

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d1 = json.load(f)
l1 = get_loads(d1.get('benchmarks', []))

with open('check_s104.json') as f:
    d2 = json.load(f)
l2 = get_loads(d2['result']['records'])

print(f"{'Load':<15} {'S081 us':<12} {'S104 us':<12} {'Diff us':<12} {'S081 sp':<12} {'S104 sp':<12}")
print("="*75)
for k in l1:
    t1, b1, sp1 = l1[k]
    t2, b2, sp2 = l2[k]
    print(f"{k:<15} {t1:<12.2f} {t2:<12.2f} {t2-t1:<12.2f} {sp1:<12.4f} {sp2:<12.4f}")
