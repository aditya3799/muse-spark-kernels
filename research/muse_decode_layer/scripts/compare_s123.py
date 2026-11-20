import json

def get_recs(fname):
    with open(fname) as f:
        d = json.load(f)
    r = d.get('result', d)
    return [x for x in r.get('records', []) if x.get('event') == 'benchmark']

r122 = get_recs('check_s122.json')
r123 = get_recs('check_s123.json')

header = f"{'Case':<22} | {'Check S122':<10} | {'Check S123':<10} | {'Diff 123-122':<12}"
print(header)
print('-' * len(header))

sum122, sum123 = 0, 0
for i in range(len(r123)):
    c = r123[i]['case']
    name = f"{c['load']} {c['layer']}"
    t122 = r122[i]['median_ns'] / 1000.0 if r122 else 0
    t123 = r123[i]['median_ns'] / 1000.0
    sum122 += t122
    sum123 += t123
    print(f"{name:<22} | {t122:10.2f} | {t123:10.2f} | {t123-t122:+10.2f} us")

print('-' * len(header))
print(f"{'SUM':<22} | {sum122:10.2f} | {sum123:10.2f} | {sum123-sum122:+10.2f} us")
