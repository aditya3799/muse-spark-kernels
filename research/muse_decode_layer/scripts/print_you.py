import json, subprocess

raw = subprocess.check_output('wsl -d Ubuntu bash -lc "~/.okc/bin/okc leaderboard muse_decode_layer --json"', shell=True)
d = json.loads(raw)
print("You:", d.get('you'))
