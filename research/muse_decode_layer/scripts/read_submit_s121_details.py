import json

with open('submit_s121.json') as f:
    d = json.load(f)

res = d.get('result', {})
print("Submit S121:")
print(f"  Score: {res.get('score')} us")
print(f"  Ratio: {res.get('ratio')}x")

recs = [r for r in res.get('records', []) if r.get('event') == 'benchmark']
for i, r in enumerate(recs):
    c = r['case']
    name = f"{c['load']} {c['layer']}"
    med = r['median_ns'] / 1000.0
    print(f"  Case {i:2d}: {name:<24} = {med:8.2f} us")
