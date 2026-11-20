import json

d108 = json.load(open("submit_s108.json"))
d126_c = json.load(open("check_s126.json"))
d126_s = json.load(open("submit_s126.json"))

def get_data(d):
    res = {}
    for r in d["result"]["records"]:
        if r.get("event") == "benchmark":
            c = r["case"]
            name = c["load"] + " " + c["layer"]
            res[name] = {
                "cand": r["median_ns"] / 1000.0,
                "base": r.get("baseline_ns", 0) / 1000.0,
                "ratio": r.get("speedup", 0.0),
            }
    return res

m108 = get_data(d108)
m126_c = get_data(d126_c)
m126_s = get_data(d126_s)

print(f"{'Shape':<22} | {'S108 cand':>10} | {'126chk cand':>11} | {'126sub cand':>11} | {'S108 base':>10} | {'126sub base':>11}")
print("-" * 88)
for k in m108:
    print(f"{k:<22} | {m108[k]['cand']:10.2f} | {m126_c[k]['cand']:11.2f} | {m126_s[k]['cand']:11.2f} | {m108[k]['base']:10.2f} | {m126_s[k]['base']:11.2f}")
print("-" * 88)
c108_sum = sum(v['cand'] for v in m108.values())
c126c_sum = sum(v['cand'] for v in m126_c.values())
c126s_sum = sum(v['cand'] for v in m126_s.values())
b108_sum = sum(v['base'] for v in m108.values())
b126s_sum = sum(v['base'] for v in m126_s.values())
print(f"{'Total':<22} | {c108_sum:10.2f} | {c126c_sum:11.2f} | {c126s_sum:11.2f} | {b108_sum:10.2f} | {b126s_sum:11.2f}")
