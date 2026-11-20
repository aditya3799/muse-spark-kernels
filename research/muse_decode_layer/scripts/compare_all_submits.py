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
r120, s120, sp120 = get_recs('submit_s120.json')
r121, s121, sp121 = get_recs('submit_s121.json')
r122, s122, sp122 = get_recs('submit_s122.json')

header = f"{'Case':<22} | {'S108':<8} | {'S120':<8} | {'S121':<8} | {'S122':<8} | {'Diff 122-108':<12}"
print(header)
print('-' * len(header))

sum108, sum120, sum121, sum122 = 0, 0, 0, 0

for i in range(len(r122)):
    c = r122[i]['case']
    name = f"{c['load']} {c['layer']}"
    t108 = r108[i]['median_ns'] / 1000.0 if r108 else 0
    t120 = r120[i]['median_ns'] / 1000.0 if r120 else 0
    t121 = r121[i]['median_ns'] / 1000.0 if r121 else 0
    t122 = r122[i]['median_ns'] / 1000.0
    sum108 += t108
    sum120 += t120
    sum121 += t121
    sum122 += t122
    print(f"{name:<22} | {t108:8.2f} | {t120:8.2f} | {t121:8.2f} | {t122:8.2f} | {t122-t108:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<22} | {sum108:8.2f} | {sum120:8.2f} | {sum121:8.2f} | {sum122:8.2f} | {sum122-sum108:+10.2f} us")
print(f"{'Official Score (us)':<22} | {s108 or 0:8.2f} | {s120 or 0:8.2f} | {s121 or 0:8.2f} | {s122 or 0:8.2f}")
print(f"{'Official Speedup':<22} | {sp108 or 0:8.4f} | {sp120 or 0:8.4f} | {sp121 or 0:8.4f} | {sp122 or 0:8.4f}")
