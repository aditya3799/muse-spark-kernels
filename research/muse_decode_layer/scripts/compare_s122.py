import json
import math
from collections import defaultdict

def get_b(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
        if 'result' in d and 'records' in d['result']:
            return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
        if 'records' in d:
            return [r for r in d['records'] if r.get('event') == 'benchmark']
    except Exception as e:
        print(f"Error loading {fname}: {e}")
    return []

b108 = get_b('profiles/submission_s108_att_cgemv.d29c356373ca40a0.json')
b121 = get_b('check_s121.json')
b122 = get_b('check_s122.json')

if not b122:
    print("No benchmark records found in check_s122.json")
    exit(1)

header = f"{'Shape':<24} | {'S108':<8} | {'S121':<8} | {'S122':<8} | {'Diff 122-121':<12} | {'Diff 122-108':<12}"
print(header)
print('-' * len(header))
tot108, tot121, tot122 = 0, 0, 0
loads121 = defaultdict(float)
loads122 = defaultdict(float)
weights = {'sliding': 46, 'full': 16}

for i in range(len(b122)):
    c = b122[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = b108[i]['median_ns']/1000.0 if i < len(b108) else 0
    t121 = b121[i]['median_ns']/1000.0 if i < len(b121) else 0
    t122 = b122[i]['median_ns']/1000.0
    tot108 += t108
    tot121 += t121
    tot122 += t122
    w = weights.get(c['layer'], 1)
    loads121[c['load']] += t121 * w
    loads122[c['load']] += t122 * w
    print(f"{name:<24} | {t108:<8.2f} | {t121:<8.2f} | {t122:<8.2f} | {t122-t121:+10.2f} us | {t122-t108:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<24} | {tot108:<8.2f} | {tot121:<8.2f} | {tot122:<8.2f} | {tot122-tot121:+10.2f} us | {tot122-tot108:+10.2f} us")

geomean121 = math.exp(sum(math.log(v) for v in loads121.values()) / len(loads121))
geomean122 = math.exp(sum(math.log(v) for v in loads122.values()) / len(loads122))
print(f"\nWeighted Load Geomean:")
print(f"  S121 Check: {geomean121:.2f} us")
print(f"  S122 Check: {geomean122:.2f} us ({geomean122 - geomean121:+0.2f} us)")

# Also print overall score and ratio if present
with open('check_s122.json') as f:
    d = json.load(f)
score = d.get('score') or (d.get('result') or {}).get('score')
ratio = d.get('ratio') or (d.get('result') or {}).get('ratio')
passed = d.get('passed') or (d.get('result') or {}).get('passed')
print(f"\nOverall Check S122:")
print(f"  Passed: {passed}")
print(f"  Score: {score} us")
print(f"  Ratio: {ratio}x")
