import json, math

d108 = json.load(open('submit_s108.json'))
d123 = json.load(open('submit_s123.json'))
d125 = json.load(open('submit_s125.json'))

def get_cand_and_base(d):
    cand = {}
    base = {}
    for r in d['result']['records']:
        if r.get('event') == 'benchmark':
            name = r['case']['load'] + ' ' + r['case']['layer']
            cand[name] = r['median_ns'] / 1000.0
            base[name] = r['baseline_ns'] / 1000.0
    return cand, base

c108, b108 = get_cand_and_base(d108)
c123, b123 = get_cand_and_base(d123)
c125, b125 = get_cand_and_base(d125)

# Projected hybrid:
# For decode sliding & verify 1 sliding: take S108
# For verify 16 & verify 32: take S123 (proven -37.5 us)
# For decode full (L=1): take S125 (proven -11.5 us)
proj = {}
for k in c108:
    if 'verify 16' in k or 'verify 32' in k:
        proj[k] = c123[k]
    elif 'full' in k and 'decode' in k:
        proj[k] = min(c108[k], c125[k])
    else:
        proj[k] = min(c108[k], c125[k])

print(f"{'Shape':<25} | {'S108 (us)':>10} | {'S123 (us)':>10} | {'S125 (us)':>10} | {'Projected (us)':>14}")
print("-" * 80)
for k in c108:
    print(f"{k:<25} | {c108[k]:10.2f} | {c123[k]:10.2f} | {c125[k]:10.2f} | {proj[k]:14.2f}")
print("-" * 80)
sum108 = sum(c108.values())
sum123 = sum(c123.values())
sum125 = sum(c125.values())
sumproj = sum(proj.values())
print(f"{'Candidate Sum (us)':<25} | {sum108:10.2f} | {sum123:10.2f} | {sum125:10.2f} | {sumproj:14.2f}")
print(f"Total Candidate Improvement vs S108: {sumproj - sum108:+.2f} us!")

# Compute projected official score using S108 baselines:
loads = {}
for k, t in proj.items():
    load, layer = k.rsplit(' ', 1)
    w = 46 if layer == 'sliding' else 16
    loads.setdefault(load, []).append((w, t, b108[k]))

load_sps = {}
for load, items in loads.items():
    t_tot = sum(w * t for w, t, _ in items)
    b_tot = sum(w * b for w, _, b in items)
    load_sps[load] = b_tot / t_tot

geomean_sp = math.exp(sum(math.log(v) for v in load_sps.values()) / len(load_sps))
# baseline geomean
base_geomean = math.exp(sum(math.log(sum(w * b for w, _, b in items)) for items in loads.values()) / len(loads))
# score us:
score_us = 9971.95 * (2.2684 / geomean_sp)
print(f"\nProjected Official Speedup: {geomean_sp:.4f}x")
print(f"Projected Official Score:   {score_us:.2f} us (vs 9,971.95 us)")
