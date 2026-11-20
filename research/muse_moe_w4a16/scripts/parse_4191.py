import json

with open("profiles/4191ab8a54cf439f.json") as f:
    data = json.load(f)

print(f"Overall score: {data['result']['score_us']:.2f} us, speedup: {data['result']['speedup']:.4f}x")
for r in data['result']['records']:
    if r.get('event') == 'benchmark':
        case = r['case']
        med = r['median_ns'] / 1000.0
        sol = r.get('speed_of_light_ns', 0) / 1000.0
        sp = r.get('speedup_of_medians', 0)
        print(f"M={case['bs']:3d}: median={med:7.2f} us (SOL={sol:6.2f} us, speedup={sp:.3f}x)")
