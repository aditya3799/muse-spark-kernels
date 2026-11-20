import json

def get_recs(fname):
    with open(fname) as f:
        d = json.load(f)
    r = d.get('result', d)
    return [x for x in r.get('records', []) if x.get('event') == 'benchmark']

r122 = get_recs('check_s122.json')
r123 = get_recs('check_s123.json')

header = f"{'Case':<22} | {'Base 122':<10} | {'Base 123':<10} | {'Diff Base':<10} | {'Cand 122':<10} | {'Cand 123':<10} | {'Diff Cand':<10}"
print(header)
print('-' * 95)

for i in range(len(r123)):
    c = r123[i]['case']
    name = f"{c['load']} {c['layer']}"
    b122 = r122[i]['baseline_ns'] / 1000.0 if r122 else 0
    b123 = r123[i]['baseline_ns'] / 1000.0
    t122 = r122[i]['median_ns'] / 1000.0 if r122 else 0
    t123 = r123[i]['median_ns'] / 1000.0
    print(f"{name:<22} | {b122:10.2f} | {b123:10.2f} | {b123-b122:+10.2f} | {t122:10.2f} | {t123:10.2f} | {t123-t122:+10.2f}")
