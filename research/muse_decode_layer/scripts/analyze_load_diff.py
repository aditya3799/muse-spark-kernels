import json
import math
from collections import defaultdict

def get_b(fname):
    with open(fname) as f:
        d = json.load(f)
    if 'result' in d and 'records' in d['result']:
        return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
    if 'records' in d:
        return [r for r in d['records'] if r.get('event') == 'benchmark']
    return []

b121 = get_b('check_s121.json')
b122 = get_b('check_s122.json')

weights = {'sliding': 46, 'full': 16}
loads121 = defaultdict(float)
loads122 = defaultdict(float)

for i in range(len(b122)):
    c = b122[i]['case']
    t121 = b121[i]['median_ns']/1000.0
    t122 = b122[i]['median_ns']/1000.0
    w = weights.get(c['layer'], 1)
    loads121[c['load']] += t121 * w
    loads122[c['load']] += t122 * w

print(f"{'Load':<15} | {'S121 (us)':<12} | {'S122 (us)':<12} | {'Diff (us)':<10}")
print('-' * 55)
for k in loads122:
    v121 = loads121[k]
    v122 = loads122[k]
    print(f"{k:<15} | {v121:<12.2f} | {v122:<12.2f} | {v122-v121:+10.2f}")

g121 = math.exp(sum(math.log(v) for v in loads121.values()) / len(loads121))
g122 = math.exp(sum(math.log(v) for v in loads122.values()) / len(loads122))
print('-' * 55)
print(f"{'Geomean':<15} | {g121:<12.2f} | {g122:<12.2f} | {g122-g121:+10.2f}")
