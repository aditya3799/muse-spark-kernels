import json
import subprocess

out = subprocess.check_output(['/home/aditya/.okc/bin/okc', 'leaderboard', 'muse_moe_w4a16', '--json'])
data = json.loads(out)
print(f"Total ranked: {data['ranked']}, You: {data['you']}")
print("\n--- Milestones ---")
for r in data['rows']:
    rk = r.get('rank')
    if rk in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 106]:
        print(f"Rank {rk:3d}: {r.get('team'):15s} speedup={r.get('speedup'):.4f}x ({r.get('value'):.2f} us)")
