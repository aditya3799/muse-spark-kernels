import json

with open('check_s104.json') as f:
    d = json.load(f)

res = d.get('result', {})
records = res.get('records', [])

events = {}
for r in records:
    ev = r.get('event')
    events.setdefault(ev, []).append(r)

print("Event types:", list(events.keys()))
for ev, recs in events.items():
    print(f"  {ev}: {len(recs)} records")
    if ev == 'benchmark':
        for b in recs[:5]:
            print("   ", b.get('case'), b.get('median_ns'), b.get('speedup'))
