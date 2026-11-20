import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('check_s108.json') as f2, open('profiles/submission_s090_s081_cgemv_m1.eccb9672e2ec40e0.json') as f3:
    d1 = json.load(f1)
    d2 = json.load(f2)
    d3 = json.load(f3)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']
b3 = d3.get('benchmarks', [])

print(f"{'Shape':<35} {'S081 (us)':<11} {'S090 (us)':<11} {'S108 (us)':<11} {'Diff S108-S081':<15}")
print("="*85)

for i in range(len(b2)):
    c = b2[i]['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t1 = b1[i]['median_ns'] / 1000.0 if i < len(b1) else 0.0
    t3 = b3[i]['median_ns'] / 1000.0 if i < len(b3) else 0.0
    t2 = b2[i]['median_ns'] / 1000.0
    print(f"{name:<35} {t1:<11.2f} {t3:<11.2f} {t2:<11.2f} {t2-t1:<15.2f}")
