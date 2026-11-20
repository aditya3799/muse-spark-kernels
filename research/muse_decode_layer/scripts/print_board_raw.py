import subprocess
import json

cmd = "/home/aditya/.okc/bin/okc leaderboard muse_decode_layer --json"
res = subprocess.run(["wsl", "-d", "Ubuntu", "bash", "-c", cmd], capture_output=True, text=True)
if res.returncode == 0:
    data = json.loads(res.stdout)
    print("Keys:", list(data.keys()))
    if "error" in data:
        print("Error in json:", data["error"])
    else:
        print("You:", data.get("you"))
        print("Entries count:", len(data.get("entries", [])))
        for e in data.get("entries", [])[:10]:
            print(f"  Rank {e.get('rank')}: {e.get('team')} - {e.get('speedup')}x ({e.get('score_us')} us)")
else:
    print("Returncode != 0:", res.stderr)
