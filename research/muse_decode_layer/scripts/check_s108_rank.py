import json
import subprocess

res = subprocess.run(['/home/aditya/.okc/bin/okc', 'leaderboard', 'muse_decode_layer', '--json'], capture_output=True, text=True)
data = json.loads(res.stdout)

rows = data.get('rows', [])
target_sp = 2.2784

for i, row in enumerate(rows):
    if row.get('speedup', 0) < target_sp:
        print(f"S108 places at rank: {row.get('rank')}")
        print(f"Ahead of: {rows[i].get('team')} ({rows[i].get('speedup')}x, {rows[i].get('value')} us)")
        if i > 0:
            print(f"Behind: {rows[i-1].get('team')} ({rows[i-1].get('speedup')}x, {rows[i-1].get('value')} us)")
        break
