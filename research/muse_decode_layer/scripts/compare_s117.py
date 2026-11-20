import json

def get_b(fname):
    with open(fname) as f:
        d = json.load(f)
    if 'result' in d and 'records' in d['result']:
        return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
    if 'records' in d:
        return [r for r in d['records'] if r.get('event') == 'benchmark']
    if 'benchmarks' in d:
        return d['benchmarks']
    return []

b108 = get_b('profiles/submission.c3d4cf6955bf4cdc.json')
b114 = get_b('check_s114.json')
b117 = get_b('check_s117.json')

header = f"{'Shape':<24} | {'S108':<8} | {'S114':<8} | {'S117':<8} | {'Diff 117-108':<12} | {'Diff 117-114':<12}"
print(header)
print('-' * len(header))
for i in range(len(b117)):
    c = b117[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = b108[i]['median_ns']/1000.0 if i < len(b108) else 0
    t114 = b114[i]['median_ns']/1000.0 if i < len(b114) else 0
    t117 = b117[i]['median_ns']/1000.0
    print(f"{name:<24} | {t108:<8.2f} | {t114:<8.2f} | {t117:<8.2f} | {t117-t108:+10.2f} us | {t117-t114:+10.2f} us")
