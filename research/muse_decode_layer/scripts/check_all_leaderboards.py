import subprocess
import json

for pb in ['muse_decode_layer', 'muse_moe_w4a16', 'muse_attention_decode', 'muse_bf16_proj']:
    cmd = f"/home/aditya/.okc/bin/okc leaderboard {pb} --json"
    res = subprocess.run(["wsl", "-d", "Ubuntu", "bash", "-c", cmd], capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        print(f"=== {pb} ===")
        print(f"You: rank {data.get('you',{}).get('rank')}, speedup {data.get('you',{}).get('speedup')}")
        entries = data.get('entries', [])[:5]
        for e in entries:
            print(f"  Rank {e.get('rank')}: {e.get('team')} - {e.get('speedup'):.4f}x ({e.get('score_us'):.2f} us)")
    else:
        print(f"Error fetching {pb}: {res.stderr[:200]}")
