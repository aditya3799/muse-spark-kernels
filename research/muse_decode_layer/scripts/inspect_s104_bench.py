import json

with open('check_s104.json') as f:
    d = json.load(f)

print("Top level keys:", list(d.keys()))
res = d.get('result', {})
print("Result keys:", list(res.keys()))
print("Result score_us:", res.get('score_us'))
print("Result speedup:", res.get('speedup'))
print("Result error:", res.get('error'))

benchmarks = res.get('benchmarks') or d.get('benchmarks')
if benchmarks:
    print("\nBENCHMARKS:")
    print(f"{'Shape':<50} {'Median us':<12} {'Speedup':<12}")
    print("="*75)
    for b in benchmarks:
        c = b['case']
        name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
        t = b.get('median_ns', 0) / 1000.0
        sp = b.get('speedup', 0)
        print(f"{name:<50} {t:<12.2f} {sp:<12.4f}")
