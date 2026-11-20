import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s103_rq_cgemv.dd00c7b9e1ba408c.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

for idx in [6, 7]:
    c = d1['profiles'][idx]['case']
    print("="*60)
    print(f"CASE {idx}: {c.get('load')} {c.get('layer')} m={c.get('batch')}")
    print("S081:")
    print(d1['profiles'][idx]['report'])
    print("S103:")
    print(d2['profiles'][idx]['report'])
