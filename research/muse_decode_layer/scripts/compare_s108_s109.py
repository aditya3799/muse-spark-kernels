import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f1, open('check_s109.json') as f2, open('profiles/submission.725c19cbeb9b4d48.json') as f0:
    d1 = json.load(f1)
    d2 = json.load(f2)
    d0 = json.load(f0)

b0 = d0.get('benchmarks', [])
b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']

print(f"{'Shape':<32} {'S081 (us)':<10} {'S108 (us)':<10} {'S109 (us)':<10} {'Diff S109-S108':<15}")
print("="*82)

for i in range(len(b2)):
    c = b2[i]['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t0 = b0[i]['median_ns'] / 1000.0 if i < len(b0) else 0.0
    t1 = b1[i]['median_ns'] / 1000.0 if i < len(b1) else 0.0
    t2 = b2[i]['median_ns'] / 1000.0
    print(f"{name:<32} {t0:<10.2f} {t1:<10.2f} {t2:<10.2f} {t2-t1:<15.2f}")
