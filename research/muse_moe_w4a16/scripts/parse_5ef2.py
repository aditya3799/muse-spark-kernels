import json

with open("profiles/submission.5ef21f3407f34944.json") as f:
    d = json.load(f)

print("Summary:")
for k, v in d.get('summary', {}).items():
    print(f"  {k}: {v}")

print("\nBenchmarks:")
for b in d.get('benchmarks', []):
    bs = b.get('batch_size') or b.get('case', {}).get('bs')
    med = b.get('median_us') or (b.get('median_ns', 0) / 1000.0)
    sp = b.get('speedup') or b.get('speedup_of_medians', 0)
    print(f"  M={bs:3d}: median={med:7.2f} us (speedup={sp:.3f}x)")
