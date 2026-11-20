import subprocess
import json

out = subprocess.check_output(["/home/aditya/.okc/bin/okc", "leaderboard", "muse_moe_w4a16", "--json"])
data = json.loads(out)
for r in data['rows']:
    rk = r.get('rank')
    if rk and 70 <= rk <= 105:
        print(f"Rank {rk:3d}: {r['team']:20s} {r['value']:8.2f} us ({r['speedup']:.4f}x)")
