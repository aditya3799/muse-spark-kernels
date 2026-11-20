import json
import math
from collections import defaultdict

with open('submit_s122.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

res = d.get('result', d)
job = res.get('job', 'unknown')
score = res.get('score')
ratio = res.get('ratio')
passed = res.get('passed')

print("=" * 60)
print(f"OFFICIAL SUBMISSION S122 RESULT")
print(f"Job: {job}")
print(f"Passed: {passed}")
print(f"Official Score: {score} us")
print(f"Official Ratio: {ratio}x")
print("=" * 60)

recs = [r for r in res.get('records', []) if r.get('event') == 'benchmark']
weights = {'sliding': 46, 'full': 16}
loads = defaultdict(float)
tot_time = 0

print(f"{'Case':<24} | {'Median (us)':<12}")
print("-" * 40)
for r in recs:
    c = r['case']
    name = f"{c['load']} {c['layer']}"
    med = r['median_ns'] / 1000.0
    tot_time += med
    w = weights.get(c['layer'], 1)
    loads[c['load']] += med * w
    print(f"{name:<24} | {med:10.2f} us")

print("-" * 40)
print(f"{'Total Raw Sum':<24} | {tot_time:10.2f} us")

g = math.exp(sum(math.log(v) for v in loads.values()) / len(loads))
print(f"{'Weighted Load Geomean':<24} | {g:10.2f} us")
