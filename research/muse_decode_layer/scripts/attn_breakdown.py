import json

with open('profiles/submission.725c19cbeb9b4d48.json') as f:
    d = json.load(f)

print(f"{'Shape':<35} {'Attn us':<12} {'Total shape us':<15} {'Attn Share':<12}")
print("="*75)

total_attn = 0.0
for p in d.get('profiles', []):
    c = p['case']
    name = f"{c.get('load')} {c.get('layer')} m={c.get('batch')}"
    attn_t = 0.0
    tot_t = 0.0
    for line in p['report'].splitlines():
        parts = line.strip().split()
        if 'attn_kernel' in line:
            for i, part in enumerate(parts):
                if part == 'µs':
                    attn_t = float(parts[i-1])
                    break
        if line.startswith('total'):
            tot_t = float(parts[1])
    total_attn += attn_t
    print(f"{name:<35} {attn_t:<12.2f} {tot_t:<15.2f} {attn_t/tot_t*100 if tot_t else 0:<12.1f}%")

print("="*75)
print(f"Total Attention time: {total_attn:.2f} us")
