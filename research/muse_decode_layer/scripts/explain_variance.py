import json

d108 = json.load(open("submit_s108.json"))
d126 = json.load(open("submit_s126.json"))
d127 = json.load(open("submit_s127.json"))

def get_stats(d):
    cand, base = {}, {}
    for r in d["result"]["records"]:
        if r.get("event") == "benchmark":
            name = r["case"]["load"] + " " + r["case"]["layer"]
            cand[name] = r["median_ns"] / 1000.0
            base[name] = r.get("baseline_ns", 0) / 1000.0
    return cand, base

c108, b108 = get_stats(d108)
c126, b126 = get_stats(d126)
c127, b127 = get_stats(d127)

print(f"{'Shape':<22} | {'S108 cand':>10} | {'S126 cand':>10} | {'S127 cand':>10} | {'S108 base':>10} | {'S126 base':>10} | {'S127 base':>10}")
print("-" * 92)
for k in c108:
    print(f"{k:<22} | {c108[k]:10.2f} | {c126[k]:10.2f} | {c127[k]:10.2f} | {b108[k]:10.2f} | {b126[k]:10.2f} | {b127[k]:10.2f}")
print("-" * 92)
print(f"{'TOTAL SUM':<22} | {sum(c108.values()):10.2f} | {sum(c126.values()):10.2f} | {sum(c127.values()):10.2f} | {sum(b108.values()):10.2f} | {sum(b126.values()):10.2f} | {sum(b127.values()):10.2f}")
