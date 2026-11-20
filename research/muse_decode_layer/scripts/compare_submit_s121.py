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

s108 = get_b('submit_s108.json')
s120 = get_b('submit_s120.json')
s121 = get_b('submit_s121.json')

header = f"{'Shape':<24} | {'S108 Sub':<10} | {'S120 Sub':<10} | {'S121 Sub':<10} | {'Diff 121-120':<12} | {'Diff 121-108':<12}"
print(header)
print('-' * len(header))
tot108, tot120, tot121 = 0, 0, 0
loads108 = defaultdict(float)
loads120 = defaultdict(float)
loads121 = defaultdict(float)
weights = {'sliding': 46, 'full': 16}

for i in range(len(s121)):
    c = s121[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = s108[i]['median_ns']/1000.0 if i < len(s108) else 0
    t120 = s120[i]['median_ns']/1000.0 if i < len(s120) else 0
    t121 = s121[i]['median_ns']/1000.0
    tot108 += t108
    tot120 += t120
    tot121 += t121
    w = weights.get(c['layer'], 1)
    loads108[c['load']] += t108 * w
    loads120[c['load']] += t120 * w
    loads121[c['load']] += t121 * w
    print(f"{name:<24} | {t108:<10.2f} | {t120:<10.2f} | {t121:<10.2f} | {t121-t120:+10.2f} us | {t121-t108:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<24} | {tot108:<10.2f} | {tot120:<10.2f} | {tot121:<10.2f} | {tot121-tot120:+10.2f} us | {tot121-tot108:+10.2f} us")

geomean108 = math.exp(sum(math.log(v) for v in loads108.values()) / len(loads108))
geomean120 = math.exp(sum(math.log(v) for v in loads120.values()) / len(loads120))
geomean121 = math.exp(sum(math.log(v) for v in loads121.values()) / len(loads121))
print(f"\nWeighted Load Geomean:")
print(f"  S108 Submit: {geomean108:.2f} us")
print(f"  S120 Submit: {geomean120:.2f} us")
print(f"  S121 Submit: {geomean121:.2f} us ({geomean121 - geomean120:+0.2f} us vs S120)")
