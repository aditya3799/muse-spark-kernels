import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s083_attn_l2_64b.dea493e561094930.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1.get('benchmarks', [])
b2 = d2.get('benchmarks', [])

print(f"{'Shape':<35} {'S081 (us)':<12} {'S083 (us)':<12} {'Diff (us)':<12}")
print("="*65)
for i in range(len(b1)):
    name = f"{b1[i]['case']['load']} {b1[i]['case']['layer']}"
    t1 = b1[i]['median_ns'] / 1000.0
    t2 = b2[i]['median_ns'] / 1000.0
    print(f"{name:<35} {t1:<12.2f} {t2:<12.2f} {t2-t1:<12.2f}")
