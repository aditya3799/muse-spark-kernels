import glob, re

def parse_profile(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    sections = re.split(r'profile \d+:', content)
    profiles = {}
    for i, sec in enumerate(sections[1:]):
        lines = [l.strip() for l in sec.strip().splitlines() if l.strip()]
        header = lines[0] if lines else ''
        kernels = {}
        for l in lines[2:]:
            if l.startswith('total'):
                kernels['total'] = float(l.split()[1].replace('µs', ''))
                break
            parts = l.split()
            # kernel name is up to the second to last parts
            # e.g. "void okc::moe_fused... 22.3 µs 31.4% 1"
            for idx, p in enumerate(parts):
                if p == 'µs':
                    time_us = float(parts[idx-1])
                    kname = " ".join(parts[:idx-1])
                    kernels[kname] = time_us
                    break
        profiles[i] = (header, kernels)
    return profiles

p114_files = glob.glob('profiles/submission_s114_*.txt')
p115_files = glob.glob('profiles/submission_s115_*.txt')

if p114_files and p115_files:
    p114 = parse_profile(p114_files[0])
    p115 = parse_profile(p115_files[0])
    print(f"Loaded p114 ({p114_files[0]}) and p115 ({p115_files[0]})")
    for i in range(min(len(p114), len(p115))):
        h114, k114 = p114[i]
        h115, k115 = p115[i]
        t114 = k114.get('total', 0)
        t115 = k115.get('total', 0)
        print(f"Shape {i}: {h114[:45]:<45} | S114: {t114:6.1f} µs | S115: {t115:6.1f} µs | Diff: {t115-t114:+6.1f} µs")
