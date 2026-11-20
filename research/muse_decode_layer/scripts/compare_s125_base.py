import json

def get_recs(fname):
    with open(fname) as f:
        d = json.load(f)
    r = d.get('result', d)
    return [x for x in r.get('records', []) if x.get('event') == 'benchmark']

r122 = get_recs('check_s122.json')
r125 = get_recs('check_s125.json')

header = f"{'Case':<22} | {'Base 122':<10} | {'Base 125':<10} | {'Diff Base':<10} | {'Cand 122':<10} | {'Cand 125':<10} | {'Diff Cand':<10}"
print(header)
print('-' * 95)

base122_sum = 0
base125_sum = 0
cand122_sum = 0
cand125_sum = 0

for i in range(len(r125)):
    c = r125[i]['case']
    name = f"{c['load']} {c['layer']}"
    b122 = r122[i]['baseline_ns'] / 1000.0 if r122 else 0
    b125 = r125[i]['baseline_ns'] / 1000.0
    t122 = r122[i]['median_ns'] / 1000.0 if r122 else 0
    t125 = r125[i]['median_ns'] / 1000.0
    base122_sum += b122
    base125_sum += b125
    cand122_sum += t122
    cand125_sum += t125
    print(f"{name:<22} | {b122:10.2f} | {b125:10.2f} | {b125-b122:+10.2f} | {t122:10.2f} | {t125:10.2f} | {t125-t122:+10.2f}")

print('-' * 95)
print(f"{'SUM':<22} | {base122_sum:10.2f} | {base125_sum:10.2f} | {base125_sum-base122_sum:+10.2f} | {cand122_sum:10.2f} | {cand125_sum:10.2f} | {cand125_sum-cand122_sum:+10.2f}")
