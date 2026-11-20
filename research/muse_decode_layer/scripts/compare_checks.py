import json

def read_records(p):
    with open(p) as f:
        d = json.load(f)
    return d["result"]["records"]

r122 = read_records("check_s122.json")
sub122 = read_records("submit_s122.json")

print(f"{'Shape':20s} | {'Check S122':10s} | {'Submit S122':11s} | {'Baseline':10s} | {'Speedup':8s}")
print("-" * 70)
total_chk = 0
total_sub = 0
for c, s in zip(r122, sub122):
    name = f"{c['mode']} {c['batch_size']} {c['seq_len_kv']}"
    chk_t = c['benchmark_time_ms'] * 1000.0
    sub_t = s['benchmark_time_ms'] * 1000.0
    base_t = s['baseline_time_ms'] * 1000.0
    total_chk += chk_t
    total_sub += sub_t
    sp = base_t / sub_t
    print(f"{name:20s} | {chk_t:7.2f} us | {sub_t:8.2f} us | {base_t:7.2f} us | {sp:6.3f}x")

print("-" * 70)
print(f"{'Raw Sum':20s} | {total_chk:7.2f} us | {total_sub:8.2f} us")
