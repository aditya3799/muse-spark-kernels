import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1:
    d1 = json.load(f1)
b1 = d1['benchmarks'][5]
for k, v in b1.items():
    if k not in ('runs_ns', 'baseline_runs_ns', 'parent_runs_ns', 'ratios'):
        print(f"{k}: {v}")
