import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f1, open('check_s114.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']

print(f"{'Shape':<28} {'S108 (us)':<10} {'S114 (us)':<10} {'Diff S114-S108':<15}")
print("="*65)

for i in range(len(b2)):
    c = b2[i]['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t1 = b1[i]['median_ns'] / 1000.0 if i < len(b1) else 0.0
    t2 = b2[i]['median_ns'] / 1000.0
    print(f"{name:<28} {t1:<10.2f} {t2:<10.2f} {t2-t1:<15.2f}")
