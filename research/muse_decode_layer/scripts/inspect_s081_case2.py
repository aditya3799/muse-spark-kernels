import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

for idx in [2, 3, 4, 5]:
    p = d['profiles'][idx]
    c = p['case']
    print("="*60)
    print(f"CASE {idx}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print(p['report'])
