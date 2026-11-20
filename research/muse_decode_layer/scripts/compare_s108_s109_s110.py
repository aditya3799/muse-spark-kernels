import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f1, open('check_s109.json') as f2, open('check_s110.json') as f3:
    d1 = json.load(f1)
    d2 = json.load(f2)
    d3 = json.load(f3)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']
b3 = [r for r in d3['result']['records'] if r.get('event') == 'benchmark']

print(f"{'Shape':<28} {'S108 (us)':<10} {'S109 (us)':<10} {'S110 (us)':<10} {'Diff S110-S108':<15} {'Diff S110-S109':<15}")
print("="*90)

for i in range(len(b3)):
    c = b3[i]['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t1 = b1[i]['median_ns'] / 1000.0 if i < len(b1) else 0.0
    t2 = b2[i]['median_ns'] / 1000.0 if i < len(b2) else 0.0
    t3 = b3[i]['median_ns'] / 1000.0
    print(f"{name:<28} {t1:<10.2f} {t2:<10.2f} {t3:<10.2f} {t3-t1:<15.2f} {t3-t2:<15.2f}")
