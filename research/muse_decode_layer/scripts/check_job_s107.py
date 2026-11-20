import subprocess
import json

res = subprocess.run(['/home/aditya/.okc/bin/okc', 'status', 'a8079ee62e6c4142', '--json'], capture_output=True, text=True)
data = json.loads(res.stdout)
for k, v in data.items():
    if k != 'result':
        print(f"{k}: {v}")
if data.get('result'):
    for k, v in data['result'].items():
        if k not in ('records', 'benchmarks', 'profiles'):
            print(f"  result.{k}: {v}")
