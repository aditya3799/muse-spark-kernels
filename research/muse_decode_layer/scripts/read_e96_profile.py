import json

path = "profiles/submission.e96a91e65df74be4.json"
with open(path) as f:
    pj = json.load(f)

print(f"Keys in {path}:", list(pj.keys()))
if "profiles" in pj:
    for i, p in enumerate(pj["profiles"]):
        c = p.get("case", {})
        print(f"Profile {i}: {c.get('load')} {c.get('layer')}")
        for l in p.get("report", "").splitlines()[:6]:
            print("  ", l)
