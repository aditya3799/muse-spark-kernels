import json
d108 = json.load(open('submit_s108.json'))['result']['records']
d120 = json.load(open('submit_s120.json'))['result']['records']

b108 = [r for r in d108 if r.get('event') == 'benchmark']
b120 = [r for r in d120 if r.get('event') == 'benchmark']

header = f"{'Shape':<24} | {'S108 base':<10} | {'S120 base':<10} | {'Base diff':<10}"
print(header)
print('-' * len(header))
for i in range(len(b108)):
    c = b108[i]['case']
    name = f"{c['load']} {c['layer']}"
    base108 = b108[i]['baseline_ns'] / 1000.0
    base120 = b120[i]['baseline_ns'] / 1000.0
    diff = base120 - base108
    print(f"{name:<24} | {base108:<10.2f} | {base120:<10.2f} | {diff:+10.2f} us")
