with open('EXPERIMENTS.md') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'S102' in l or 'S103' in l:
        print(f"Line {i+1}: {l.strip()[:100]}")
