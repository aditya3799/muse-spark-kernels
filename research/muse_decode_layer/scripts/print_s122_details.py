import json

def load_res(p):
    with open(p) as f:
        return json.load(f)

s122_check = load_res("check_s122.json")
s122_sub = load_res("submit_s122.json")

print("=== CHECK S122 ===")
for r in s122_check["results"]:
    name = f"{r['mode']} {r['batch_size']} {r['seq_len_kv']}"
    med = r["benchmark"]["median"] * 1e6
    base = r["baseline"]["median"] * 1e6
    ratio = base / med
    print(f"{name:20s}: {med:7.2f} us (base {base:7.2f} us -> {ratio:.4f}x)")

print("\n=== SUBMIT S122 ===")
for r in s122_sub["results"]:
    name = f"{r['mode']} {r['batch_size']} {r['seq_len_kv']}"
    med = r["benchmark"]["median"] * 1e6
    base = r["baseline"]["median"] * 1e6
    ratio = base / med
    print(f"{name:20s}: {med:7.2f} us (base {base:7.2f} us -> {ratio:.4f}x)")
