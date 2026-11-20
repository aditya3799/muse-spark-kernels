import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('check_s108.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1.get('benchmarks', [])
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']

print(f"{'Shape':<35} {'Base S081 (us)':<15} {'Base S108 (us)':<15} {'Ratio':<10}")
print("="*75)
for i in range(len(b1)):
    name = f"{b1[i]['case']['load']} {b1[i]['case']['layer']}"
    base1 = b1[i]['baseline_ns'] / 1000.0
    base2 = b2[i]['baseline_ns'] / 1000.0
    print(f"{name:<35} {base1:<15.2f} {base2:<15.2f} {base2/base1:<10.4f}")
