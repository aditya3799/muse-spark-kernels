import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f:
    d = json.load(f)

print(f"{'Shape':<35} {'Median (us)':<12} {'Speedup':<10}")
print("="*60)
for b in d.get('benchmarks', []):
    c = b['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    t = b['median_ns'] / 1000.0
    sp = b['speedup']
    print(f"{name:<35} {t:<12.2f} {sp:<10.4f}")
