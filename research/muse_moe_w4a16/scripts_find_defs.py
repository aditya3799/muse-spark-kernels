with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v172_hybrid_small.py') as f:
    lines = f.readlines()

for i, l in enumerate(lines[3900:]):
    if l.startswith('def ') or 'defines' in l.lower() or 'dispatch' in l.lower() or 'moe' in l.lower():
        print(f"{3901+i}: {l.strip()[:100]}")
