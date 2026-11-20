import json

with open('check_s104.json') as f:
    d = json.load(f)

res = d.get('result', {})
profiles = [r for r in res.get('records', []) if r.get('event') == 'profile']

for idx in [2, 3, 4, 5]:
    p = profiles[idx]
    c = p['case']
    print("="*60)
    print(f"CASE {idx}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print(p['report'])
