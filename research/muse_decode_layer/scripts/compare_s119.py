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

b108 = get_b('profiles/submission_s108_att_cgemv.d29c356373ca40a0.json')
b114 = get_b('check_s114.json')
b119 = get_b('check_s119.json')

header = f"{'Shape':<24} | {'S108':<8} | {'S114':<8} | {'S119':<8} | {'Diff 119-114':<12} | {'Diff 119-108':<12}"
print(header)
print('-' * len(header))
sum108, sum114, sum119 = 0, 0, 0
for i in range(len(b114)):
    c = b114[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = b108[i]['median_ns']/1000.0 if i < len(b108) else 0
    t114 = b114[i]['median_ns']/1000.0
    t119 = b119[i]['median_ns']/1000.0 if i < len(b119) else 0
    sum108 += t108
    sum114 += t114
    sum119 += t119
    print(f"{name:<24} | {t108:<8.2f} | {t114:<8.2f} | {t119:<8.2f} | {t119-t114:+10.2f} us | {t119-t108:+10.2f} us")
print('-' * len(header))
print(f"{'SUM':<24} | {sum108:<8.2f} | {sum114:<8.2f} | {sum119:<8.2f} | {sum119-sum114:+10.2f} us | {sum119-sum108:+10.2f} us")
