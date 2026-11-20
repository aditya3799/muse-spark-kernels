import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s103_rq_cgemv.dd00c7b9e1ba408c.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

for k in d1:
    if k not in ('benchmarks', 'profiles'):
        print(f"d1 {k}: {d1[k]}")
for k in d2:
    if k not in ('benchmarks', 'profiles'):
        print(f"d2 {k}: {d2[k]}")
