import json

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/profiles/submission_v190_optimal_hybrid.ccc3114620374d8d.json') as f:
    d = json.load(f)

print("Score:", d.get('score_us'))
print("Speedup:", d.get('speedup'))
print("Passed:", d.get('passed'))
print("Benchmarks count:", len(d.get('benchmarks', [])))
for b in d.get('benchmarks', [])[:5]:
    print(b.get('case'), b.get('median_ns')/1000.0, "us, sp:", b.get('speedup'))
