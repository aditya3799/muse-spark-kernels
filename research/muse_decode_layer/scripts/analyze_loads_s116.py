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
l116 = analyze('check_s116.json')

print(f"{'Load':<15} | {'S108':<12} | {'S114':<12} | {'S116':<12} | {'116/108':<10}")
print('-' * 70)
for k in sorted(l108.keys()):
    r116 = l116[k] / l108[k]
    print(f"{k:<15} | {l108[k]:<12.1f} | {l114[k]:<12.1f} | {l116[k]:<12.1f} | {r116:<10.4f}")

gm_116_108 = math.exp(sum(math.log(l116[k] / l108[k]) for k in l108) / len(l108))
print('-' * 70)
print(f"Geomean S116 / S108: {gm_116_108:.4f} (relative speedup: {1.0/gm_116_108:.4f}x)")
