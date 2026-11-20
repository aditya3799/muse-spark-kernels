import json

with open("profiles/submission.5ef21f3407f34944.json") as f:
    d = json.load(f)

for b in d['benchmarks']:
    bs = b['case']['bs']
    med = b['median_ns'] / 1000.0
    base = b['baseline_ns'] / 1000.0
    sp = b['speedup']
    print(f"M={bs:3d}: sglang_baseline={base:7.2f} us, our_median={med:7.2f} us, speedup={sp:.4f}x")
