import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f1, open('profiles/submission_s090_s081_cgemv_m1.eccb9672e2ec40e0.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)

p1 = d1['profiles'][5]
p2 = d2['profiles'][5]

print("=== S081 (725c19cb, 275.86 us) ===")
print(p1['report'])
print("\n=== S090 (eccb9672, 296.10 us) ===")
print(p2['report'])
