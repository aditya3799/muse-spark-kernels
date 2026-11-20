import json
import re

with open('profiles/submission_s122_att_qdir_pk2.880c66b81bff42e1.json', encoding='utf-8') as f:
    d = json.load(f)

for idx, p in enumerate(d['profiles']):
    c = p['case']
    name = f"{c['load']} {c['layer']}"
    report = p['report']
    print(f"\nCase {idx}: {name}")
    for line in report.splitlines():
        if any(term in line for term in ['sk2', 'pool', 'router', 'cgemv', 'rq', 'nvjet']):
            print(" ", line.strip())
