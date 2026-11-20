import json

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
f109 = 'check_s109.json'
f114 = 'check_s114.json'
f115 = 'check_s115.json'

print(f'S108 score: {get_score("submit_s108.json")}')
print(f'S109 score: {get_score(f109)}')
print(f'S114 score: {get_score(f114)}')
print(f'S115 score: {get_score(f115)}')

b108 = get_benchmarks(f108)
b109 = get_benchmarks(f109)
b114 = get_benchmarks(f114)
b115 = get_benchmarks(f115)

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

header = f"{'Shape':<28} | {'S108':<8} | {'S109':<8} | {'S114':<8} | {'S115':<8} | {'Diff 115-108':<12}"
print(header)
print('-' * len(header))

for i in range(len(b115)):
    name = get_name(b115[i])
    t108 = get_time(b108[i]) if i < len(b108) else 0.0
    t109 = get_time(b109[i]) if i < len(b109) else 0.0
    t114 = get_time(b114[i]) if i < len(b114) else 0.0
    t115 = get_time(b115[i])
    diff = t115 - t108
    print(f"{name:<28} | {t108:<8.2f} | {t109:<8.2f} | {t114:<8.2f} | {t115:<8.2f} | {diff:+10.2f} us")
