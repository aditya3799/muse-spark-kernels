import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('check_s104.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

res2 = d2.get('result', {})
benchmarks2 = [r for r in res2.get('records', []) if r.get('event') == 'benchmark']
benchmarks1 = d1.get('benchmarks', [])

print(f"{'Shape':<45} {'S081 (us)':<12} {'S104 (us)':<12} {'Diff (us)':<12} {'S081 sp':<12} {'S104 sp':<12}")
print("="*105)

for i, rec in enumerate(benchmarks2):
    c = rec['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t1 = benchmarks1[i].get('median_ns', 0) / 1000.0 if i < len(benchmarks1) else 0.0
    sp1 = benchmarks1[i].get('speedup', 0) if i < len(benchmarks1) else 0.0
    t2 = rec.get('median_ns', 0) / 1000.0
    sp2 = rec.get('speedup', 0)
    diff = t2 - t1
    print(f"{name:<45} {t1:<12.2f} {t2:<12.2f} {diff:<12.2f} {sp1:<12.4f} {sp2:<12.4f}")
