import json

def get_b(fname):
    with open(fname) as f:
        d = json.load(f)
    if 'result' in d and 'records' in d['result']:
        return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
    if 'records' in d:
        return [r for r in d['records'] if r.get('event') == 'benchmark']
    return []

s108 = get_b('submit_s108.json')
s120 = get_b('submit_s120.json')

header = f"{'Shape':<24} | {'Submit S108':<12} | {'Submit S120':<12} | {'Diff 120-108':<14}"
print(header)
print('-' * len(header))
tot108, tot120 = 0, 0
for i in range(len(s108)):
    c = s108[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = s108[i]['median_ns']/1000.0
    t120 = s120[i]['median_ns']/1000.0 if i < len(s120) else 0
    tot108 += t108
    tot120 += t120
    print(f"{name:<24} | {t108:<12.2f} | {t120:<12.2f} | {t120-t108:+12.2f} us")
print('-' * len(header))
print(f"{'SUM':<24} | {tot108:<12.2f} | {tot120:<12.2f} | {tot120-tot108:+12.2f} us")
