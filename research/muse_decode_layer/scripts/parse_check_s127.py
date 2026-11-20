import json

with open("check_s127.json") as f:
    d = json.load(f)

res = d.get("result", {})
print("Status:", res.get("status"))
print("Passed:", res.get("passed"))
print("Flags:", res.get("flags"))
print("Score (us):", res.get("score_us"))
print("Speedup:", res.get("speedup"))

# Print individual benchmarks
cands = {}
for r in res.get("records", []):
    if r.get("event") == "benchmark":
        c = r["case"]
        name = c["load"] + " " + c["layer"]
        cands[name] = {
            "cand_us": r["median_ns"] / 1000.0,
            "base_us": r.get("baseline_ns", 0) / 1000.0,
            "ratio": r.get("speedup", 0.0)
        }

print(f"\n{'Shape':<25} | {'Candidate (us)':>15} | {'Baseline (us)':>15} | {'Speedup':>10}")
print("-" * 75)
for k, v in cands.items():
    print(f"{k:<25} | {v['cand_us']:15.2f} | {v['base_us']:15.2f} | {v['ratio']:10.4f}x")
print("-" * 75)
cand_sum = sum(v['cand_us'] for v in cands.values())
print(f"{'Total Candidate Sum':<25} | {cand_sum:15.2f}")
