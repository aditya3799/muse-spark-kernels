import json

with open('profiles/submission_s121_mid_wsi_wpburst.5bae49b074ea4191.json', encoding='utf-8') as f:
    d121 = json.load(f)

with open('profiles/submission_s122_att_qdir_pk2.880c66b81bff42e1.json', encoding='utf-8') as f:
    d122 = json.load(f)

p1 = d121['profiles'][2]
p2 = d122['profiles'][2]

print("=== Case 2: S121 ===")
print(p1['report'])

print("\n=== Case 2: S122 ===")
print(p2['report'])
