import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

for b in d.get('benchmarks', []):
    c = b['case']
    if c.get('batch') == 32:
        print(f"batch=32 load={c.get('load')} layer={c.get('layer')} median_us={b['median_ns']/1000.0:.2f} mean_us={b['mean_ns']/1000.0:.2f} best_us={b['best_ns']/1000.0:.2f} worst_us={b['worst_ns']/1000.0:.2f}")
