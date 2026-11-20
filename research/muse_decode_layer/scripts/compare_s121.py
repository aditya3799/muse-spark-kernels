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

b108 = get_b('profiles/submission_s108_att_cgemv.d29c356373ca40a0.json')
b120 = get_b('check_s120.json')
b121 = get_b('check_s121.json')

header = f"{'Shape':<24} | {'S108':<8} | {'S120':<8} | {'S121':<8} | {'Diff 121-120':<12} | {'Diff 121-108':<12}"
print(header)
print('-' * len(header))
tot108, tot120, tot121 = 0, 0, 0
loads120 = defaultdict(float)
loads121 = defaultdict(float)
weights = {'sliding': 46, 'full': 16}

for i in range(len(b121)):
    c = b121[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = b108[i]['median_ns']/1000.0 if i < len(b108) else 0
    t120 = b120[i]['median_ns']/1000.0 if i < len(b120) else 0
    t121 = b121[i]['median_ns']/1000.0
    tot108 += t108
    tot120 += t120
    tot121 += t121
    w = weights.get(c['layer'], 1)
    loads120[c['load']] += t120 * w
    loads121[c['load']] += t121 * w
    print(f"{name:<24} | {t108:<8.2f} | {t120:<8.2f} | {t121:<8.2f} | {t121-t120:+10.2f} us | {t121-t108:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<24} | {tot108:<8.2f} | {tot120:<8.2f} | {tot121:<8.2f} | {tot121-tot120:+10.2f} us | {tot121-tot108:+10.2f} us")

geomean120 = math.exp(sum(math.log(v) for v in loads120.values()) / len(loads120))
geomean121 = math.exp(sum(math.log(v) for v in loads121.values()) / len(loads121))
print(f"\nWeighted Load Geomean:")
print(f"  S120 Check: {geomean120:.2f} us")
print(f"  S121 Check: {geomean121:.2f} us ({geomean121 - geomean120:+0.2f} us)")
