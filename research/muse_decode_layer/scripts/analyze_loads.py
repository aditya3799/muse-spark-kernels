import json, math

def analyze(fname):
    with open(fname) as f:
        d = json.load(f)
    if 'result' in d and 'records' in d['result']:
        records = [r for r in d['result']['records'] if r.get('event') == 'benchmark']
    elif 'records' in d:
        records = [r for r in d['records'] if r.get('event') == 'benchmark']
    elif 'benchmarks' in d:
        records = d['benchmarks']
    else:
        records = []
    loads = {}
    for r in records:
        load = r['case']['load']
        w = r['case']['weight']
        t = r['median_ns'] / 1000.0
        loads[load] = loads.get(load, 0.0) + w * t
    return loads

l108 = analyze('profiles/submission.c3d4cf6955bf4cdc.json')
l114 = analyze('check_s114.json')
l115 = analyze('check_s115.json')

print(f"{'Load':<15} | {'S108':<12} | {'S114':<12} | {'S115':<12} | {'114/108':<10} | {'115/108':<10}")
print('-' * 80)
for k in sorted(l108.keys()):
    r114 = l114[k] / l108[k]
    r115 = l115[k] / l108[k]
    print(f"{k:<15} | {l108[k]:<12.1f} | {l114[k]:<12.1f} | {l115[k]:<12.1f} | {r114:<10.4f} | {r115:<10.4f}")

gm_114_108 = math.exp(sum(math.log(l114[k] / l108[k]) for k in l108) / len(l108))
gm_115_108 = math.exp(sum(math.log(l115[k] / l108[k]) for k in l108) / len(l108))
print('-' * 80)
print(f"Geomean S114 / S108: {gm_114_108:.4f} (relative speedup: {1.0/gm_114_108:.4f}x)")
print(f"Geomean S115 / S108: {gm_115_108:.4f} (relative speedup: {1.0/gm_115_108:.4f}x)")
