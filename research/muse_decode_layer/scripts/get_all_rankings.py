import subprocess
import json

problems = [
    "muse_decode_layer",
    "muse_moe_w4a16",
    "muse_attention_decode",
    "muse_bf16_proj",
    "muse_prefill_layer",
]

print("=" * 65)
print(f"{'Board / Problem':<25} | {'Rank':<6} | {'Total':<6} | {'Speedup':<8}")
print("-" * 65)

# 1. Overall
cmd = ["wsl", "-d", "Ubuntu", "bash", "-lc", "~/.okc/bin/okc leaderboard --json"]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if p.returncode == 0:
    d = json.loads(p.stdout)
    y = d.get("you", {})
    tot = d.get("ranked", 0)
    print(f"{'OVERALL':<25} | {y.get('rank', '-'):<6} | {tot:<6} | {'N/A':<8}")

for prob in problems:
    cmd = ["wsl", "-d", "Ubuntu", "bash", "-lc", f"~/.okc/bin/okc leaderboard {prob} --json"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        d = json.loads(p.stdout)
        y = d.get("you", {})
        tot = d.get("ranked", 0)
        sp = f"{y.get('speedup'):.4f}x" if y.get('speedup') is not None else "-"
        print(f"{prob:<25} | {y.get('rank', '-'):<6} | {tot:<6} | {sp:<8}")

print("=" * 65)
