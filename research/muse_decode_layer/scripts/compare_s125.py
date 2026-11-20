import json
import os

path = "check_s125.json"
if not os.path.exists(path) or os.path.getsize(path) == 0:
    print(f"{path} not ready yet")
    exit(0)

def get_recs(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
        r = d.get('result', d)
        recs = [x for x in r.get('records', []) if x.get('event') == 'benchmark']
        return recs, r.get('score_us'), r.get('speedup'), r.get('passed'), r.get('flags')
    except Exception as e:
        return None, None, None, None, None

r122, s122, sp122, p122, f122 = get_recs('check_s122.json')
r123, s123, sp123, p123, f123 = get_recs('check_s123.json')
r125, s125, sp125, p125, f125 = get_recs('check_s125.json')

print("=== S125 CHECK STATUS ===")
print("Passed:", p125)
print("Flags:", f125)
print("Score (us):", s125)
print("Speedup:", sp125)

if r125:
    header = f"{'Case':<22} | {'S122':<8} | {'S123':<8} | {'S125':<8} | {'Diff 125-122':<12}"
    print(header)
    print('-' * len(header))
    sum122, sum123, sum125 = 0, 0, 0
    for i in range(len(r125)):
        c = r125[i]['case']
        name = f"{c['load']} {c['layer']}"
        t122 = r122[i]['median_ns'] / 1000.0 if r122 else 0
        t123 = r123[i]['median_ns'] / 1000.0 if r123 else 0
        t125 = r125[i]['median_ns'] / 1000.0
        sum122 += t122
        sum123 += t123
        sum125 += t125
        print(f"{name:<22} | {t122:8.2f} | {t123:8.2f} | {t125:8.2f} | {t125-t122:+10.2f} us")
    print('-' * len(header))
    print(f"{'SUM':<22} | {sum122:8.2f} | {sum123:8.2f} | {sum125:8.2f} | {sum125-sum122:+10.2f} us")
