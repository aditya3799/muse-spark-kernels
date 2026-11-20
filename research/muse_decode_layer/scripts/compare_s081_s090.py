import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s090_s081_cgemv_m1.eccb9672e2ec40e0.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

print("S081 score:", d1.get('summary'))
print("S090 score:", d2.get('summary'))

b1 = d1.get('benchmarks', [])
b2 = d2.get('benchmarks', [])

print(f"{'Shape':<50} {'S081 (us)':<12} {'S090 (us)':<12} {'Diff (us)':<12}")
print("="*80)
for i in range(len(b1)):
    c = b1[i]['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t1 = b1[i].get('median_ns', 0) / 1000.0
    t2 = b2[i].get('median_ns', 0) / 1000.0
    diff = t2 - t1
    print(f"{name:<50} {t1:<12.2f} {t2:<12.2f} {diff:<12.2f}")
