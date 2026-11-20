import json

with open("check_s122.json") as f:
    d = json.load(f)
print("Keys in check_s122.json:", list(d.keys()))
if "run" in d:
    print("Keys in d['run']:", list(d["run"].keys()))
    if "results" in d["run"]:
        for r in d["run"]["results"][:3]:
            print(r.get("name"), r.get("benchmarks"))
