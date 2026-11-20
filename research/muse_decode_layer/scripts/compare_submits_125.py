import json

def get_recs(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
        r = d.get('result', d)
        recs = [x for x in r.get('records', []) if x.get('event') == 'benchmark']
        return recs, r.get('score_us'), r.get('speedup')
    except Exception as e:
        return None, None, None

r108, s108, sp108 = get_recs('submit_s108.json')
r122, s122, sp122 = get_recs('submit_s122.json')
r123, s123, sp123 = get_recs('submit_s123.json')
r125, s125, sp125 = get_recs('submit_s125.json')

header = f"{'Case':<22} | {'S108':<8} | {'S122':<8} | {'S123':<8} | {'S125':<8} | {'Diff 125-108':<12}"
print(header)
print('-' * len(header))

sum108, sum122, sum123, sum125 = 0, 0, 0, 0

for i in range(len(r125)):
    c = r125[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = r108[i]['median_ns'] / 1000.0 if r108 else 0
    t122 = r122[i]['median_ns'] / 1000.0 if r122 else 0
    t123 = r123[i]['median_ns'] / 1000.0 if r123 else 0
    t125 = r125[i]['median_ns'] / 1000.0
    sum108 += t108
    sum122 += t122
    sum123 += t123
    sum125 += t125
    print(f"{name:<22} | {t108:8.2f} | {t122:8.2f} | {t123:8.2f} | {t125:8.2f} | {t125-t108:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<22} | {sum108:8.2f} | {sum122:8.2f} | {sum123:8.2f} | {sum125:8.2f} | {sum125-sum108:+10.2f} us")
print(f"{'Official Score (us)':<22} | {s108 or 0:8.2f} | {s122 or 0:8.2f} | {s123 or 0:8.2f} | {s125 or 0:8.2f}")
print(f"{'Official Speedup':<22} | {sp108 or 0:8.4f} | {sp122 or 0:8.4f} | {sp123 or 0:8.4f} | {sp125 or 0:8.4f}")
