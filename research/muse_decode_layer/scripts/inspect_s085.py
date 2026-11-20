import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s085_q1_occ3.6ea69b33f4d747a3.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

print("S081 score:", d1.get('score_us'), "sp:", d1.get('speedup'))
print("S085 score:", d2.get('score_us'), "sp:", d2.get('speedup'))

b1 = d1.get('benchmarks', [])
b2 = d2.get('benchmarks', [])

print(f"{'Shape':<35} {'S081 (us)':<12} {'S085 (us)':<12} {'Diff (us)':<12}")
print("="*65)
for i in range(len(b1)):
    name = f"{b1[i]['case']['load']} {b1[i]['case']['layer']}"
    t1 = b1[i]['median_ns'] / 1000.0
    t2 = b2[i]['median_ns'] / 1000.0
    print(f"{name:<35} {t1:<12.2f} {t2:<12.2f} {t2-t1:<12.2f}")
