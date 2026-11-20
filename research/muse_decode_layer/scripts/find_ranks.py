import json, subprocess

raw = subprocess.check_output('wsl -d Ubuntu bash -lc "~/.okc/bin/okc leaderboard muse_decode_layer --json"', shell=True)
data = json.loads(raw)
for r in data['rows']:
    sp = r.get('speedup')
    if sp and 2.27 <= sp <= 2.32:
        val = r.get('value', 0)
        print(f"Rank {r.get('rank'):>4}: {sp:.4f}x ({val:.2f} us) - {r.get('team')}")
