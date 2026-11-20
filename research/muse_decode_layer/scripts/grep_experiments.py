with open('EXPERIMENTS.md') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'B4FIN' in l or 'S052' in l or 'S060' in l:
        print(f"Line {i+1}: {l.strip()[:100]}")
