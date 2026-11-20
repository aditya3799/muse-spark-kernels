import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

for i, p in enumerate(d.get('profiles', [])):
    c = p['case']
    sk2_lines = [l.strip() for l in p['report'].splitlines() if 'sk2_kernel' in l]
    if sk2_lines:
        print(f"Case {i}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
        for l in sk2_lines:
            print("  ", l)
