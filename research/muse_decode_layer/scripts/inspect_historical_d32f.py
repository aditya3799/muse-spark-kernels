import json
import glob
import os

for path in sorted(glob.glob('profiles/*.json')):
    try:
        with open(path) as f:
            d = json.load(f)
        benchmarks = d.get('benchmarks', [])
        for b in benchmarks:
            c = b.get('case', {})
            if c.get('load') == 'decode 32' and c.get('layer') == 'full':
                print(f"{os.path.basename(path):<30} median_us={b.get('median_ns')/1000.0:.2f} sp={b.get('speedup'):.4f}")
    except Exception as e:
        pass
