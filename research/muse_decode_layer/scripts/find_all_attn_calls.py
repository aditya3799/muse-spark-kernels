import json

with open('profiles/submission_s122_att_qdir_pk2.880c66b81bff42e1.json', encoding='utf-8') as f:
    d = json.load(f)

for p in d['profiles']:
    c = p['case']
    name = f"{c['load']} {c['layer']}"
    for line in p['report'].splitlines():
        if "attn" in line:
            print(f"{name:22s} | {line.strip()}")
