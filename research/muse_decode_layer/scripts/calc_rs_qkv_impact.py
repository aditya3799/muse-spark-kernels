import json

with open('profiles/submission.c3d4cf6955bf4cdc.json') as f:
    d = json.load(f)

benchmarks = d.get('benchmarks', [])
# Calculate baseline and current times
total_cur = 0.0
total_opt = 0.0

savings = {
    ('decode 32', 'sliding'): 11.0,
    ('decode 32', 'full'): 10.0,
    ('decode 16', 'sliding'): 5.85,
    ('decode 16', 'full'): 5.55,
}

print(f"{'Shape':<28} {'Current (us)':<12} {'Optimized (us)':<15} {'Saving (us)':<12}")
print("="*70)

for b in benchmarks:
    c = b['case']
    name = f"{c.get('load')} {c.get('layer')}"
    m = c.get('batch')
    cur_us = b['median_ns'] / 1000.0
    key = (f"{c.get('load')}", c.get('layer'))
    save = savings.get(key, 0.0)
    opt_us = cur_us - save
    total_cur += cur_us
    total_opt += opt_us
    print(f"{name:<28} {cur_us:<12.2f} {opt_us:<15.2f} {save:<12.2f}")

print("="*70)
print(f"Total current latency:   {total_cur:.2f} us")
print(f"Total optimized latency: {total_opt:.2f} us")
print(f"Raw latency reduction:   {total_cur - total_opt:.2f} us")
sglang_score = d.get('score', {}).get('baseline_us', 22620.0)
cur_score = d.get('score', {}).get('score_us', 9971.95)
cur_speedup = sglang_score / cur_score
opt_score = cur_score * (total_opt / total_cur) # approx
opt_speedup = sglang_score / opt_score
print(f"Current Speedup: {cur_speedup:.4f}x (score: {cur_score:.2f} us)")
print(f"Estimated Speedup: {opt_speedup:.4f}x (score: {opt_score:.2f} us)")
