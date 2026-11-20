import json

d108 = json.load(open('submit_s108.json'))
d123 = json.load(open('submit_s123.json'))
d125 = json.load(open('submit_s125.json'))

def get_cand(d):
    return {r['case']['load'] + ' ' + r['case']['layer']: r['median_ns']/1000.0 for r in d['result']['records'] if r.get('event') == 'benchmark'}

c108 = get_cand(d108)
c123 = get_cand(d123)
c125 = get_cand(d125)

print(f"{'Shape':<25} | {'S108':>8} | {'S123':>8} | {'S125':>8} | {'125 vs 123':>10}")
print("-" * 65)
for k in c108:
    diff = c125[k] - c123[k]
    print(f"{k:<25} | {c108[k]:8.2f} | {c123[k]:8.2f} | {c125[k]:8.2f} | {diff:+10.2f}")
