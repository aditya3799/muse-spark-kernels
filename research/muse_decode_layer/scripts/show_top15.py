import json

d = json.load(open('leaderboard_muse_decode_layer.json'))
print(f"{'Rank':<6} {'Team':<25} {'Speedup':<10} {'Score (us)':<12}")
print("="*55)
for r in d.get('rows', [])[:20]:
    val = r.get('value', 0)
    print(f"{r.get('rank'):<6} {r.get('team'):<25} {r.get('speedup'):<10.4f} {val:<12.2f}")

you = d.get('you', {})
print("="*55)
print(f"YOU: Rank {you.get('rank')}, Speedup {you.get('speedup'):.4f}x")
