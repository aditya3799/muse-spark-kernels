import json

def parse_profile(path):
    d = json.load(open(path))
    prof = d.get('profile', '')
    return prof

p121 = parse_profile('submit_s121.json')
p122 = parse_profile('submit_s122.json')

print("S121 profile length:", len(p121))
print("S122 profile length:", len(p122))

# Let's see the cases in S122 profile
lines = p122.splitlines()
case = None
for l in lines:
    if "Case" in l or "decode" in l or "verify" in l:
        print(l[:80])
