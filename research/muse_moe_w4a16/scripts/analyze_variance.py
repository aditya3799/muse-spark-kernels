import json

with open("profiles/4191ab8a54cf439f.json") as f:
    d = json.load(f)

for b in d['benchmarks']:
    bs = b['case']['bs']
    runs = [x / 1000.0 for x in b['runs_ns']]
    runs.sort()
    p10 = runs[len(runs)//10]
    p50 = runs[len(runs)//2]
    p90 = runs[int(len(runs)*0.9)]
    print(f"M={bs:3d}: min={min(runs):.2f} us, p10={p10:.2f} us, p50={p50:.2f} us, p90={p90:.2f} us, max={max(runs):.2f} us, mean={sum(runs)/len(runs):.2f} us")
