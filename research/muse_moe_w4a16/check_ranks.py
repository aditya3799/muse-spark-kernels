import subprocess, json

output = subprocess.check_output(["/mnt/d/OpenKernelChallenge/bin/okc", "leaderboard", "muse_moe_w4a16", "--json"])
data = json.loads(output)
for row in data["rows"]:
    r = row.get("rank")
    if r is not None and r in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 101]:
        print(f"Rank {r:3d}: {row['speedup']:.4f}x ({row['value']:.1f} us) by {row['team']}")
