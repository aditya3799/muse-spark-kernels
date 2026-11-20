import json

def parse_prof_json(fname):
    with open(fname) as f:
        d = json.load(f)
    return d['profile']['json']

j121 = parse_prof_json('submit_s121.json')
j122 = parse_prof_json('submit_s122.json')

def print_case_kernels(pj, case_idx):
    c = pj['cases'][case_idx]
    print(f"Case {case_idx}: {c['name']} (total {c['gpu_busy_us']:.2f} us)")
    for k in c['kernels']:
        print(f"  {k['name'][:60]:60s} | {k['dur_us']:6.2f} us | count {k['count']}")

print("--- S121 Case 1 ---")
print_case_kernels(j121, 1)

print("\n--- S122 Case 1 ---")
print_case_kernels(j122, 1)
