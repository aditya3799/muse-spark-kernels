import subprocess
import json

cmd = ["wsl", "-d", "Ubuntu", "bash", "-lc", "~/.okc/bin/okc leaderboard muse_decode_layer --json"]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.returncode == 0:
    d = json.loads(p.stdout)
    rows = d.get("rows", [])
    for r in [1, 5, 10, 25, 50, 75, 100, 125, 150, 175]:
        if r <= len(rows):
            row = rows[r - 1]
            print(f"Rank {row['rank']:3d}: {row['team']:<18} | Speedup: {row['speedup']:.4f}x | Score: {row['value']:.2f} us")
