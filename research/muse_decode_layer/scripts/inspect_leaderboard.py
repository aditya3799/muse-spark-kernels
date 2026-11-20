import json

with open('leaderboard_muse_decode_layer.json', 'r', encoding='utf-16') as f:
    d = json.load(f)

print("You:", d.get('you'))
rows = d.get('rows', [])
print("\nTop 15:")
for r in rows[:15]:
    print(f"#{r.get('rank')}: {r.get('team')} - {r.get('speedup'):.4f}x ({r.get('value'):.2f} us)")

print("\nAround rank 176:")
for r in rows:
    if 170 <= r.get('rank', 0) <= 180:
        print(f"#{r.get('rank')}: {r.get('team')} - {r.get('speedup'):.4f}x ({r.get('value'):.2f} us)")
