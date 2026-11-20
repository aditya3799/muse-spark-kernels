import json, math

def get_benchmarks(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
            if 'result' in d and 'records' in d['result']:
                return [r for r in d['result']['records'] if r.get('event') == 'benchmark']
            if 'records' in d:
                return [r for r in d['records'] if r.get('event') == 'benchmark']
            if 'benchmarks' in d:
                return d['benchmarks']
            return []
    except Exception as e:
        return []

def get_score(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
            r = d.get('result', d)
            return r.get('score_us', 'N/A')
    except:
        return 'N/A'

f108 = 'profiles/submission.c3d4cf6955bf4cdc.json'
f114 = 'check_s114.json'
f116 = 'check_s116.json'

print(f'S108 score: {get_score("submit_s108.json")}')
print(f'S114 score: {get_score(f114)}')
print(f'S116 score: {get_score(f116)}')

b108 = get_benchmarks(f108)
b114 = get_benchmarks(f114)
b116 = get_benchmarks(f116)

def get_time(b_entry):
    if not b_entry:
        return 0.0
    if 'median_ns' in b_entry:
        return b_entry['median_ns'] / 1000.0
    if 'time_us' in b_entry:
        return b_entry['time_us']
    return 0.0

def get_name(b_entry):
    if 'case' in b_entry:
        c = b_entry['case']
        return f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    if 'name' in b_entry:
        return b_entry['name']
    return 'unknown'

header = f"{'Shape':<28} | {'S108':<8} | {'S114':<8} | {'S116':<8} | {'Diff 116-108':<12} | {'Diff 116-114':<12}"
print(header)
print('-' * len(header))

for i in range(len(b116)):
    name = get_name(b116[i])
    t108 = get_time(b108[i]) if i < len(b108) else 0.0
    t114 = get_time(b114[i]) if i < len(b114) else 0.0
    t116 = get_time(b116[i])
    diff_108 = t116 - t108
    diff_114 = t116 - t114
    print(f"{name:<28} | {t108:<8.2f} | {t114:<8.2f} | {t116:<8.2f} | {diff_108:+10.2f} us | {diff_114:+10.2f} us")
