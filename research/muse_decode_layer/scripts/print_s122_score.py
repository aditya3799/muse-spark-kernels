import json

with open('submit_s122.json') as f:
    d = json.load(f)

r = d.get('result', {})
print(f"Job ID: {d.get('id')}")
print(f"Score US: {r.get('score_us')}")
print(f"Baseline US: {r.get('baseline_us')}")
print(f"Speedup: {r.get('speedup')}")
print(f"Speedup CI: {r.get('speedup_ci')}")
print(f"Flags: {r.get('flags')}")
