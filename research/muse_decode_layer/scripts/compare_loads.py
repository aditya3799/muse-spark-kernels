import json, math

def get_b(fname):
    with open(fname) as f:
        d = json.load(f)
    if 'result' in d and 'records' in d['result']:
        return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
    if 'benchmarks' in d:
        return d['benchmarks']
    return []

b108 = get_b('profiles/submission_s108_att_cgemv.d29c356373ca40a0.json')
b114 = get_b('check_s114.json')

loads108 = {}
loads114 = {}

for b in b108:
    c = b['case']
    l = c['load']
    w = c['weight']
    t = b['median_ns'] / 1000.0
    loads108[l] = loads108.get(l, 0.0) + w * t

for b in b114:
    c = b['case']
    l = c['load']
    w = c['weight']
    t = b['median_ns'] / 1000.0
    loads114[l] = loads114.get(l, 0.0) + w * t

print("Per-load total time (46*sliding + 16*full):")
header = f"{'Load':<15} | {'S108 (us)':<12} | {'S114 (us)':<12} | {'Diff':<10}"
print(header)
print('-' * len(header))
prod108 = 1.0
prod114 = 1.0
for l in sorted(loads108.keys()):
    t0 = loads108[l]
    t1 = loads114[l]
    prod108 *= t0
    prod114 *= t1
    print(f"{l:<15} | {t0:<12.2f} | {t1:<12.2f} | {t1-t0:+10.2f} us")

geomean108 = prod108 ** (1.0 / len(loads108))
geomean114 = prod114 ** (1.0 / len(loads114))
print('-' * len(header))
print(f"{'Geometric Mean':<15} | {geomean108:<12.2f} | {geomean114:<12.2f} | {geomean114-geomean108:+10.2f} us")
