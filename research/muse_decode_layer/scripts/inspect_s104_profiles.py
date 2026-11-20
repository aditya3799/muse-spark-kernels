import json

with open('check_s104.json') as f:
    d = json.load(f)

res = d.get('result', {})
profiles = [r for r in res.get('records', []) if r.get('event') == 'profile']

for p in profiles[8:]:
    c = p['case']
    print("="*60)
    print(f"{c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print(p['report'])
