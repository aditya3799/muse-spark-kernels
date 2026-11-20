import subprocess
import json

for prob in ['muse_moe_w4a16', 'muse_attention_decode', 'muse_bf16_proj']:
    res = subprocess.run(['/home/aditya/.okc/bin/okc', 'leaderboard', prob, '--json'], capture_output=True, text=True)
    try:
        d = json.loads(res.stdout)
        rows = d.get('rows', [])[:5]
        print(f"\n=== {prob} Top 5 ===")
        for r in rows:
            print(f"Rank {r.get('rank')}: {r.get('team')} {r.get('speedup')}x ({r.get('value'):.2f} us)")
    except Exception as e:
        print(f"Error {prob}: {e}")
