import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f1, open('submit_s114.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2.get('result', d2).get('records', []) if r.get('event') == 'benchmark']

header = f"{'Shape':<28} | {'S108 (us)':<10} | {'S114 Submit (us)':<16} | {'Diff':<10}"
print(header)
print('-' * len(header))
for i in range(len(b2)):
    c = b2[i]['case']
    name = f"{c.get('load')} {c.get('layer')}"
    t1 = b1[i]['median_ns'] / 1000.0 if i < len(b1) else 0.0
    t2 = b2[i]['median_ns'] / 1000.0
    diff = t2 - t1
    print(f"{name:<28} | {t1:<10.2f} | {t2:<16.2f} | {diff:+10.2f} us")
