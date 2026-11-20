import json, subprocess

raw = subprocess.check_output('wsl -d Ubuntu bash -lc "~/.okc/bin/okc leaderboard muse_decode_layer --json"', shell=True)
data = json.loads(raw)
rows = sorted([r for r in data['rows'] if r.get('submitted_at')], key=lambda r: r['submitted_at'], reverse=True)
for r in rows[:5]:
    diff = 1793861989.262 - r['submitted_at']
    print(f"{r['team']:<20} submitted_at={r['submitted_at']:.1f} (our scored was {diff:+.1f}s relative to this)")
