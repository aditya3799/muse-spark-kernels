import json

with open("check_s126.json") as f:
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
        cands[name] = r["median_ns"] / 1000.0

print(f"\n{'Shape':<25} | {'Candidate Time (us)':>20}")
print("-" * 50)
for k, v in cands.items():
    print(f"{k:<25} | {v:20.2f}")
print("-" * 50)
print(f"{'Total Candidate Sum':<25} | {sum(cands.values()):20.2f}")
