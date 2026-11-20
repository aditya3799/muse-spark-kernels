import json

def get_recs(fname):
    with open(fname) as f:
        d = json.load(f)
    r = d.get('result', d)
    return [x for x in r.get('records', []) if x.get('event') == 'benchmark']

r108 = get_recs('submit_s108.json')
r122 = get_recs('submit_s122.json')

print(f"{'Case':<22} | {'Base S108':<10} | {'Base S122':<10} | {'Diff Base':<10} | {'Cand S108':<10} | {'Cand S122':<10} | {'Diff Cand':<10}")
print('-' * 95)
b108_sum = 0
b122_sum = 0
c108_sum = 0
c122_sum = 0
for i in range(len(r122)):
    c = r122[i]['case']
    name = f"{c['load']} {c['layer']}"
    b1 = r108[i].get('baseline_ns', 0) / 1000.0
    b2 = r122[i].get('baseline_ns', 0) / 1000.0
    c1 = r108[i]['median_ns'] / 1000.0
    c2 = r122[i]['median_ns'] / 1000.0
    b108_sum += b1
    b122_sum += b2
    c108_sum += c1
    c122_sum += c2
    print(f"{name:<22} | {b1:10.2f} | {b2:10.2f} | {b2-b1:+10.2f} | {c1:10.2f} | {c2:10.2f} | {c2-c1:+10.2f}")

print('-' * 95)
print(f"{'SUM':<22} | {b108_sum:10.2f} | {b122_sum:10.2f} | {b122_sum-b108_sum:+10.2f} | {c108_sum:10.2f} | {c122_sum:10.2f} | {c122_sum-c108_sum:+10.2f}")
