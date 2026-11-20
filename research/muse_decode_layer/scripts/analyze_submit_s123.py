import json
import os

path = "submit_s123.json"
if not os.path.exists(path) or os.path.getsize(path) == 0:
    print(f"{path} not ready yet")
    exit(0)

with open(path) as f:
    d = json.load(f)

res = d.get("result", {})
print("=== SUBMIT S123 RESULT ===")
print("Passed:", res.get("passed"))
print("Flags:", res.get("flags"))
print("Score (us):", res.get("score_us"))
print("Speedup:", res.get("speedup"))
if res.get("error"):
    print("Error:", res.get("error"))

recs = [x for x in res.get("records", []) if x.get("event") == "benchmark"]
print(f"Total benchmark shapes: {len(recs)}")
cand_sum = 0
base_sum = 0
for r in recs:
    c = r["case"]
    name = f"{c['load']} {c['layer']}"
    med = r["median_ns"] / 1000.0
    base = r["baseline_ns"] / 1000.0
    sp = base / med
    cand_sum += med
    base_sum += base
    print(f"  {name:22s} | {med:7.2f} us (base {base:7.2f} us -> {sp:.4f}x)")

print(f"Total Candidate Sum: {cand_sum:.2f} us")
print(f"Total Baseline Sum:  {base_sum:.2f} us")
print(f"Overall Ratio:       {base_sum / cand_sum:.4f}x")
