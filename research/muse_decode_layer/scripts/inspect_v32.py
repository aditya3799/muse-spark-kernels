import json

d108 = json.load(open('submit_s108.json'))
d123 = json.load(open('submit_s123.json'))
d125 = json.load(open('submit_s125.json'))

for name, d in [('S108', d108), ('S123', d123), ('S125', d125)]:
    for r in d['result']['records']:
        if r.get('event') == 'benchmark' and r['case']['load'] == 'verify 32':
            c = r['case']
            cand = r['median_ns'] / 1000.0
            base = r['baseline_ns'] / 1000.0
            print(f"{name} {c['layer']}: cand={cand:.1f}us base={base:.1f}us ratio={base/cand:.4f}")
