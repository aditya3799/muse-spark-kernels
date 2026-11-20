import subprocess
import json

cmd = "/home/aditya/.okc/bin/okc leaderboard muse_decode_layer --json"
res = subprocess.run(["wsl", "-d", "Ubuntu", "bash", "-c", cmd], capture_output=True, text=True)
if res.returncode == 0:
    data = json.loads(res.stdout)
    you = data.get("you", {})
    print(f"You: rank {you.get('rank')}, speedup {you.get('speedup'):.4f}x, score {you.get('score_us'):.2f} us")
    entries = data.get("entries", [])
    print(f"Total ranked entries: {len(entries)}")
    for e in entries:
        r = e.get("rank")
        if r in [1, 2, 3, 5, 10, 25, 50, 75, 100, 125, 150, you.get("rank")]:
            print(f"  Rank {r:3d}: {e.get('team'):20s} - {e.get('speedup'):.4f}x ({e.get('score_us'):.2f} us)")
else:
    print("Error:", res.stderr)
