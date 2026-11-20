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
b120 = get_b('check_s120.json')

loads108, loads114, loads120 = {}, {}, {}

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

for b in b120:
    c = b['case']
    l = c['load']
    w = c['weight']
    t = b['median_ns'] / 1000.0
    loads120[l] = loads120.get(l, 0.0) + w * t

header = f"{'Load':<15} | {'S108 (us)':<12} | {'S114 (us)':<12} | {'S120 (us)':<12} | {'Diff 120-108':<12}"
print(header)
print('-' * len(header))
prod108, prod114, prod120 = 1.0, 1.0, 1.0
for l in sorted(loads108.keys()):
    t0 = loads108[l]
    t1 = loads114[l]
    t2 = loads120[l]
    prod108 *= t0
    prod114 *= t1
    prod120 *= t2
    print(f"{l:<15} | {t0:<12.2f} | {t1:<12.2f} | {t2:<12.2f} | {t2-t0:+10.2f} us")

geomean108 = prod108 ** (1.0 / len(loads108))
geomean114 = prod114 ** (1.0 / len(loads114))
geomean120 = prod120 ** (1.0 / len(loads120))
print('-' * len(header))
print(f"{'Geometric Mean':<15} | {geomean108:<12.2f} | {geomean114:<12.2f} | {geomean120:<12.2f} | {geomean120-geomean108:+10.2f} us")
