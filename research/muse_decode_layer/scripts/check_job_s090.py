import subprocess
import json

out = subprocess.check_output(['/home/aditya/.okc/bin/okc', 'status', '43aec79e497c41c9', '--json'])
d = json.loads(out)
for k, v in d.items():
    if k not in ('benchmarks', 'profiles'):
        print(f"{k}: {v}")
