import json

with open('profiles/submission_s120_wsi_fpart.4541145832f0451f.json') as f:
    d = json.load(f)

for i, p in enumerate(d['profiles']):
    c = p.get('case', {})
    name = f"{c.get('load')} {c.get('layer')}"
    print(f"\n==================== Case {i}: {name} ====================")
    lines = p.get('report', '').strip().splitlines()
    for l in lines[:12]:
        print(l)
