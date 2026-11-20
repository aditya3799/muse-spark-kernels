import json

with open('check_s109.json') as f:
    d = json.load(f)

profiles = [r for r in d['result']['records'] if r.get('event') == 'profile']
for idx in [2, 3, 4, 5]:
    p = profiles[idx]
    c = p['case']
    print("="*60)
    print(f"CASE {idx}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print(p['report'])
