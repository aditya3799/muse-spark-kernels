import json

def get_prof(path, case_idx):
    d = json.load(open(path))
    recs = [x for x in d['result']['records'] if x.get('event') == 'benchmark']
    return recs[case_idx].get('profile', '')

p121 = get_prof('submit_s121.json', 2)
p122 = get_prof('submit_s122.json', 2)

print("=== Case 2 (decode 16 sliding) S121 profile ===")
print(p121[:1000])

print("\n=== Case 2 (decode 16 sliding) S122 profile ===")
print(p122[:1000])
