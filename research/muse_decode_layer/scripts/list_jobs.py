import subprocess
import json

out = subprocess.check_output(['/home/aditya/.okc/bin/okc', 'jobs', '--json'])
jobs = json.loads(out)
for j in jobs[:12]:
    print(f"{j.get('job_id')}  {j.get('problem'):<20}  {j.get('mode'):<7}  {j.get('status'):<8}  speedup={j.get('score')}  score_us={j.get('score_us')}")
