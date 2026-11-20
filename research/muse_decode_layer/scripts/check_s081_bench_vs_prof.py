import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

for i, b in enumerate(d.get('benchmarks', [])):
    p = d['profiles'][i]
    c = b['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    bench_us = b.get('median_ns', 0) / 1000.0
    prof_total = 0.0
    prof_busy = 0.0
    for line in p.get('report', '').splitlines():
        if line.startswith('total'):
            parts = line.split()
            prof_total = float(parts[1])
        if 'GPU busy' in line:
            parts = line.split()
            for part in parts:
                if part.replace('.', '', 1).isdigit():
                    prof_busy = float(part)
                    break
    print(f"{name:<40} bench={bench_us:<8.2f} prof_total={prof_total:<8.2f} prof_busy={prof_busy:<8.2f}")
