import json

with open('/mnt/d/OpenKernelChallenge/muse_bf16_proj/profiles/submission_plain_qkv1_qkv32_v187.4589211989af437c.json') as f:
    d = json.load(f)

print("v187 in bf16_proj:")
print("Passed:", d.get('passed'))
print("Score (us):", d.get('score_us'))
print("Speedup:", d.get('speedup'))
for b in d.get('benchmarks', []):
    c = b['case']
    print(f"  {c.get('layer'):<12} m={c.get('m'):<5} median_us={b.get('median_ns')/1000.0:.2f} sp={b.get('speedup'):.4f}")
