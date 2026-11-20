import json
import re

with open('profiles/submission_s121_mid_wsi_wpburst.5bae49b074ea4191.json', encoding='utf-8') as f:
    d = json.load(f)

cases = [1, 3, 5, 7, 9, 11] # full layers
for idx in cases:
    p = d['profiles'][idx]
    c = p['case']
    name = f"{c['load']} {c['layer']}"
    report = p['report']
    for line in report.splitlines():
        if 'attn_kernel' in line:
            m = re.search(r'([\d\.]+)\s*\S+s\s+([\d\.]+)%', line)
            if m:
                print(f"{name:<20}: attn = {m.group(1)} us ({m.group(2)}%)")
