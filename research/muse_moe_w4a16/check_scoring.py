import json
import math

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/profiles/submission.5f5a8c8b72b24ee4.json') as f:
    data = json.load(f)

print("Summary:", data.get('summary'))
speedups = []
times = []
baselines = []
for b in data['benchmarks']:
    c = b['case']
    t = b['median_ns'] / 1000.0
    base = b['baseline_ns'] / 1000.0
    sp = b['speedup']
    times.append(t)
    baselines.append(base)
    speedups.append(sp)
    print(f"M={c['bs']:3d}: ours={t:.2f}us, sglang={base:.2f}us, speedup={sp:.4f}x")

# Let's check geometric mean of speedups
geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
print(f"Geometric mean speedup: {geo_mean:.4f}x")
print(f"Baseline / geo_mean = {13448.5 / geo_mean:.2f} us")
