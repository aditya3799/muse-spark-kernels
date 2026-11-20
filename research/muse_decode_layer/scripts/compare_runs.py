import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s090_s081_cgemv_m1.eccb9672e2ec40e0.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

b1 = d1['benchmarks'][5]
b2 = d2['benchmarks'][5]

runs1 = [x/1000.0 for x in b1.get('runs_ns', [])]
runs2 = [x/1000.0 for x in b2.get('runs_ns', [])]

print("S081 runs:", sorted(runs1))
print("S090 runs:", sorted(runs2))
