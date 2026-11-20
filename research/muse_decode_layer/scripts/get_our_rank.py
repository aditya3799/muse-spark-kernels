import subprocess
import json

cmd = ["wsl", "-d", "Ubuntu", "bash", "-lc", "~/.okc/bin/okc leaderboard muse_decode_layer --json"]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.returncode == 0:
    d = json.loads(p.stdout)
    print("Leaderboard standing for 'you':")
    print(json.dumps(d.get("you"), indent=2))
else:
    print("Error running command:", p.stderr)
