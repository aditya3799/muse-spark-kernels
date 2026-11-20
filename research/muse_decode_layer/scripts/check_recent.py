import json, subprocess, datetime

raw = subprocess.check_output('wsl -d Ubuntu bash -lc "~/.okc/bin/okc leaderboard muse_decode_layer --json"', shell=True)
data = json.loads(raw)
rows = sorted([r for r in data['rows'] if r.get('submitted_at')], key=lambda r: r['submitted_at'], reverse=True)
print(f"{'Team':<22} {'Rank':<6} {'Speedup':<10} {'Submitted Time'}")
print("-" * 60)
for r in rows[:10]:
    t = datetime.datetime.fromtimestamp(r['submitted_at']).strftime('%H:%M:%S')
    print(f"{r.get('team', ''):<22} {str(r.get('rank', '')):<6} {r.get('speedup', 0.0):.4f}x    {t}")
