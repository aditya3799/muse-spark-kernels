import json
import re

with open('profiles/submission_s121_mid_wsi_wpburst.5bae49b074ea4191.json', encoding='utf-8') as f:
    d121 = json.load(f)

with open('profiles/submission_s122_att_qdir_pk2.880c66b81bff42e1.json', encoding='utf-8') as f:
    d122 = json.load(f)

for idx in range(len(d122['profiles'])):
    p1 = d121['profiles'][idx]
    p2 = d122['profiles'][idx]
    c = p2['case']
    name = f"{c['load']} {c['layer']}"
    
    # parse top kernels from report
    def get_kernels(report):
        k = {}
        for line in report.splitlines():
            m = re.search(r'^(.*?)\s+([\d\.]+)\s*\S+s\s+([\d\.]+)%', line)
            if m:
                kname = m.group(1).strip()
                ktime = float(m.group(2))
                k[kname] = ktime
        return k
    
    k1 = get_kernels(p1['report'])
    k2 = get_kernels(p2['report'])
    
    print(f"\n=== Case {idx}: {name} ===")
    all_k = set(k1.keys()) | set(k2.keys())
    for kname in sorted(all_k):
        t1 = k1.get(kname, 0.0)
        t2 = k2.get(kname, 0.0)
        diff = t2 - t1
        if abs(diff) > 0.5:
            print(f"  {kname:<35}: {t1:6.2f} us -> {t2:6.2f} us ({diff:+6.2f} us)")
