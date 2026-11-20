import json

with open('profiles/submission.b5f40cfa779043f3.json') as f:
    d125 = json.load(f)

for p in d125.get('profiles', []):
    c = p.get('case', {})
    if c.get('load') == 'decode 32' and c.get('layer') == 'sliding':
        print(f"=== S125 Profile for {c.get('load')} {c.get('layer')} ===")
        print(p.get('report', ''))
