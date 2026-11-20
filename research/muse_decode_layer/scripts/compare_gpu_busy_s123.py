import json
import re
from collections import defaultdict
import math

with open('check_s122.json', encoding='utf-8') as f:
    d122 = json.load(f)

with open('check_s123.json', encoding='utf-8') as f:
    d123 = json.load(f)

def get_gpu_busy(report):
    tot = 0
    for line in report.splitlines():
        if 'GPU busy' in line:
            m = re.search(r'([\d\.]+)\s*\S+s', line)
            if m:
                return float(m.group(1))
        if 'total' in line:
            m = re.search(r'total\s+([\d\.]+)\s*\S+s', line)
            if m:
                tot = float(m.group(1))
    return tot

print(f"{'Case':<22} | {'S122 GPU busy':<14} | {'S123 GPU busy':<14} | {'Diff':<10}")
print('-' * 65)

tot122_busy, tot123_busy = 0, 0
loads122 = defaultdict(float)
loads123 = defaultdict(float)
weights = {'sliding': 46, 'full': 16}

profs122 = d122.get('profile', {}).get('json', {}).get('cases', []) if isinstance(d122.get('profile'), dict) else []
profs123 = d123.get('profile', {}).get('json', {}).get('cases', []) if isinstance(d123.get('profile'), dict) else []

if not profs122:
    # check profiles field
    profs122 = d122.get('profiles', [])
    profs123 = d123.get('profiles', [])

for idx in range(min(len(profs122), len(profs123))):
    p1 = profs122[idx]
    p2 = profs123[idx]
    c = p2.get('case', {})
    name = f"{c.get('load', idx)} {c.get('layer', '')}"
    
    b1 = p1.get('gpu_busy_us', get_gpu_busy(p1.get('report', '')))
    b2 = p2.get('gpu_busy_us', get_gpu_busy(p2.get('report', '')))
    
    tot122_busy += b1
    tot123_busy += b2
    
    w = weights.get(c.get('layer'), 1)
    loads122[c.get('load', idx)] += b1 * w
    loads123[c.get('load', idx)] += b2 * w
    
    print(f"{name:<22} | {b1:10.2f} us   | {b2:10.2f} us   | {b2-b1:+8.2f} us")

print('-' * 65)
print(f"{'SUM':<22} | {tot122_busy:10.2f} us   | {tot123_busy:10.2f} us   | {tot123_busy-tot122_busy:+8.2f} us")
