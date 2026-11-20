import json

with open("check_s122.json") as f:
    d = json.load(f)
res = d.get("result", {})
print("Keys in result:", list(res.keys()))
if "shapes" in res:
    print(f"Total shapes: {len(res['shapes'])}")
    for s in res["shapes"]:
        print(s)
elif "benchmarks" in res:
    for b in res["benchmarks"]:
        print(b)
