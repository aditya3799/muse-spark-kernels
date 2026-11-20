import json

with open('submit_s122.json') as f:
    d = json.load(f)

print("Keys:", list(d.keys()))
for k in ['job', 'score', 'ratio', 'rank', 'passed', 'status']:
    if k in d:
        print(f"{k}: {d[k]}")

if 'result' in d:
    r = d['result']
    print("result keys:", list(r.keys()))
    for k in ['job', 'score', 'ratio', 'rank', 'passed', 'status']:
        if k in r:
            print(f"result.{k}: {r[k]}")
