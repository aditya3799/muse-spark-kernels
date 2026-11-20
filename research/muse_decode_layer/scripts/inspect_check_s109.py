import json

with open('check_s109.json') as f:
    d = json.load(f)

res = d.get('result', {})
print("Passed:", res.get('passed'))
print("Error:", res.get('error'))
print("Score (us):", res.get('score_us'))
print("Speedup:", res.get('speedup'))
print("Tests:", res.get('tests_passed'), "/", res.get('tests'))
records = res.get('records', [])
benchmarks = [r for r in records if r.get('event') == 'benchmark']
print(f"Total benchmarks: {len(benchmarks)}")
