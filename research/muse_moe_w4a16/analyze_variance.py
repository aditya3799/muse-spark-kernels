import json

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/profiles/submission.5f5a8c8b72b24ee4.json') as f:
    data = json.load(f)

for b in data['benchmarks']:
    c = b['case']
    times = [t / 1000.0 for t in b['runs_ns']]
    times.sort()
    p10 = times[int(len(times) * 0.1)]
    p50 = times[int(len(times) * 0.5)]
    p90 = times[int(len(times) * 0.9)]
    print(f"bs={c['bs']:3d} ({c['load']:10s}): min={times[0]:.2f}us p10={p10:.2f}us p50={p50:.2f}us p90={p90:.2f}us max={times[-1]:.2f}us std={b['std_ns']/1000.0:.2f}us sglang={b['baseline_ns']/1000.0:.2f}us speedup={b['speedup']:.3f}x")
