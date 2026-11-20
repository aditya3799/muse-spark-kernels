import json
import re
from collections import defaultdict
import math

with open('profiles/submission_s121_mid_wsi_wpburst.5bae49b074ea4191.json', encoding='utf-8') as f:
    d121 = json.load(f)

with open('profiles/submission_s122_att_qdir_pk2.880c66b81bff42e1.json', encoding='utf-8') as f:
    d122 = json.load(f)

def get_gpu_busy(report):
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

print(f"{'Case':<22} | {'S121 GPU busy':<14} | {'S122 GPU busy':<14} | {'Diff':<10}")
print('-' * 65)

tot121_busy, tot122_busy = 0, 0
loads121 = defaultdict(float)
loads122 = defaultdict(float)
weights = {'sliding': 46, 'full': 16}

for idx in range(len(d122['profiles'])):
    p1 = d121['profiles'][idx]
    p2 = d122['profiles'][idx]
    c = p2['case']
    name = f"{c['load']} {c['layer']}"
    
    b1 = get_gpu_busy(p1['report'])
    b2 = get_gpu_busy(p2['report'])
    
    tot121_busy += b1
    tot122_busy += b2
    
    w = weights.get(c['layer'], 1)
    loads121[c['load']] += b1 * w
    loads122[c['load']] += b2 * w
    
    print(f"{name:<22} | {b1:10.2f} us   | {b2:10.2f} us   | {b2-b1:+8.2f} us")

print('-' * 65)
print(f"{'SUM':<22} | {tot121_busy:10.2f} us   | {tot122_busy:10.2f} us   | {tot122_busy-tot121_busy:+8.2f} us")

g121 = math.exp(sum(math.log(v) for v in loads121.values()) / len(loads121))
g122 = math.exp(sum(math.log(v) for v in loads122.values()) / len(loads122))
print(f"\nWeighted Load Geomean on GPU Busy:")
print(f"  S121: {g121:.2f} us")
print(f"  S122: {g122:.2f} us ({g122-g121:+0.2f} us)")
