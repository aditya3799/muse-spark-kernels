import json

with open('leaderboard.json', 'w') as out:
    pass

import subprocess
res = subprocess.run(['/home/aditya/.okc/bin/okc', 'leaderboard', 'muse_decode_layer', '--json'], capture_output=True, text=True)
data = json.loads(res.stdout)

rows = data.get('rows', [])
my_score = 9945.44

for i, row in enumerate(rows):
    if row.get('value', 999999) > my_score:
        print(f"S090 would place at rank: {row.get('rank')}")
        print(f"Ahead of: {rows[i].get('team')} ({rows[i].get('speedup')}x, {rows[i].get('value')} us)")
        if i > 0:
            print(f"Behind: {rows[i-1].get('team')} ({rows[i-1].get('speedup')}x, {rows[i-1].get('value')} us)")
        break
