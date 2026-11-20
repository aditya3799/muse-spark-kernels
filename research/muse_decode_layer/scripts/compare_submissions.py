import json

with open('submit_s108.json') as f:
    d108 = json.load(f)
with open('submit_s120.json') as f:
    d120 = json.load(f)

r108 = d108['result']
r120 = d120['result']

print(f"S108 Official: speedup={r108['speedup']:.4f} score_us={r108['score_us']:.2f}")
print(f"S120 Official: speedup={r120['speedup']:.4f} score_us={r120['score_us']:.2f}")

b108 = [r for r in r108['records'] if r.get('event') == 'benchmark']
b120 = [r for r in r120['records'] if r.get('event') == 'benchmark']

header = f"{'Shape':<24} | {'S108 med':<10} | {'S120 med':<10} | {'S108 best':<10} | {'S120 best':<10} | {'Diff med':<10}"
print(header)
print('-' * len(header))
for i in range(len(b108)):
    c = b108[i]['case']
    name = f"{c['load']} {c['layer']}"
    med108 = b108[i]['median_ns'] / 1000.0
    med120 = b120[i]['median_ns'] / 1000.0
    best108 = b108[i]['best_ns'] / 1000.0
    best120 = b120[i]['best_ns'] / 1000.0
    diff = med120 - med108
    print(f"{name:<24} | {med108:<10.2f} | {med120:<10.2f} | {best108:<10.2f} | {best120:<10.2f} | {diff:+10.2f} us")
