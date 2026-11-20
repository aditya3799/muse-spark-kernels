import json

with open('submit_s108.json') as f:
    d = json.load(f)

print("ID:", d.get('id'))
print("Status:", d.get('status'))
print("Mode:", d.get('mode'))
res = d.get('result', {})
if res:
    print("Passed:", res.get('passed'))
    print("Error:", res.get('error'))
    print("Score (us):", res.get('score_us'))
    print("Speedup:", res.get('speedup'))
    print("CI:", res.get('speedup_ci'))
else:
    print("No inline result (may need okc status)")
