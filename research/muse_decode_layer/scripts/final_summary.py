import json, subprocess

raw = subprocess.check_output('wsl -d Ubuntu bash -lc "~/.okc/bin/okc leaderboard muse_decode_layer --json"', shell=True)
data = json.loads(raw)

total = data.get("ranked", 0)
you = data.get("you", {})
rank = you.get("rank")
sp = you.get("speedup")
percentile = (1.0 - (rank / total)) * 100.0

print(f"Total Ranked Teams: {total}")
print(f"Our Team: {you.get('team')}")
print(f"Final Official Rank: #{rank} of {total} (Top {100 - percentile:.1f}%, beating {percentile:.1f}% of all competitors)")
print(f"Final Official Speedup: {sp:.4f}x (9,971.95 us)")

print("\n--- TOP 10 FINAL STANDINGS ---")
for r in data.get("rows", [])[:10]:
    print(f"Rank {r.get('rank'):>2}: {r.get('team'):<20} {r.get('speedup'):.4f}x ({r.get('value'):.2f} us)")
