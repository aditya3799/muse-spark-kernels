import json

path = "profiles/submission.83e98efcdc1e4fdc.json"
try:
    with open(path) as f:
        pj = json.load(f)
    print("Keys in profile:", list(pj.keys()))
    for i, p in enumerate(pj.get("profiles", [])):
        c = p.get("case", {})
        print(f"Profile {i}: {c.get('load')} {c.get('layer')}")
        for l in p.get("report", "").splitlines()[:5]:
            print("  ", l)
except Exception as e:
    print("Error:", e)
