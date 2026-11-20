import subprocess
import json

cmd = ["wsl", "-d", "Ubuntu", "bash", "-lc", "~/.okc/bin/okc problems --json"]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.returncode == 0:
    d = json.loads(p.stdout)
    for prob in d.get("problems", []):
        if prob.get("name") == "muse_decode_layer":
            print(json.dumps(prob, indent=2))
