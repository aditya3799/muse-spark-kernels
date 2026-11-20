import json

with open('check_s108.json') as f1, open('check_s109.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = [r for r in d1['result']['records'] if r.get('event') == 'benchmark']
b2 = [r for r in d2['result']['records'] if r.get('event') == 'benchmark']

print(f"{'Shape':<32} {'Base 108 (us)':<15} {'Base 109 (us)':<15} {'Base diff':<12}")
print("="*65)
for i in range(len(b1)):
    name = f"{b1[i]['case']['load']} {b1[i]['case']['layer']}"
    base1 = b1[i]['baseline_ns'] / 1000.0
    base2 = b2[i]['baseline_ns'] / 1000.0
    print(f"{name:<32} {base1:<15.2f} {base2:<15.2f} {base2-base1:<12.2f}")
