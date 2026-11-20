import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s103_rq_cgemv.dd00c7b9e1ba408c.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

for i in range(len(d1.get('profiles', []))):
    p1 = d1['profiles'][i]
    p2 = d2['profiles'][i]
    c = p1['case']
    print(f"=== CASE {i}: {c.get('load')} {c.get('layer')} m={c.get('batch')} ===")
    print("S081 kernels:")
    for line in p1['report'].splitlines()[:6]:
        print("  ", line)
    print("S103 kernels:")
    for line in p2['report'].splitlines()[:6]:
        print("  ", line)
