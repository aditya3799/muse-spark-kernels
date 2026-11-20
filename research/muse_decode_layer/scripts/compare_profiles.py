import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s103_rq_cgemv.dd00c7b9e1ba408c.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

p1 = d1.get('profiles', [])
p2 = d2.get('profiles', [])

for idx in [2, 3, 4, 5]:
    c = p1[idx]['case']
    print("="*60)
    print(f"CASE {idx}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print("--- S081 ---")
    print(p1[idx]['report'])
    print("--- S103 ---")
    print(p2[idx]['report'])
