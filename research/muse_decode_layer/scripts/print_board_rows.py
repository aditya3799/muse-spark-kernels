import subprocess
import json

cmd = "/home/aditya/.okc/bin/okc leaderboard muse_decode_layer --json"
res = subprocess.run(["wsl", "-d", "Ubuntu", "bash", "-c", cmd], capture_output=True, text=True)
if res.returncode == 0:
    data = json.loads(res.stdout)
    you = data.get("you", {})
    print(f"You: rank {you.get('rank')}, speedup {you.get('speedup')}x")
    rows = data.get("rows", [])
    print(f"Total rows: {len(rows)}")
    targets = [1, 2, 3, 5, 10, 25, 50, 75, 100, 125, 150, 175, you.get("rank")]
    for r in rows:
        rk = r.get("rank")
        if rk in targets:
            sp = r.get('speedup')
            sc = r.get('score_us')
            sp_str = f"{sp:.4f}x" if sp is not None else "None"
            sc_str = f"{sc:.2f} us" if sc is not None else "None"
            print(f"  Rank {rk:3d}: {str(r.get('team')):22s} - {sp_str:8s} ({sc_str})")
else:
    print("Error:", res.stderr)
