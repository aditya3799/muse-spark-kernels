import subprocess
import json

res = subprocess.run(['/home/aditya/.okc/bin/okc', 'leaderboard', 'muse_decode_layer', '--json'], capture_output=True, text=True)
data = json.loads(res.stdout)
you = data.get('you', {})
print("Our team:", you.get('team'))
print("Our new rank:", you.get('rank'))
print("Our speedup:", you.get('speedup'))
print("Ahead of:", you.get('ahead'))
print("To first:", you.get('to_first'))
