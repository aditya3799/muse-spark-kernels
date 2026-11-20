with open("submission.py") as f:
    for idx, l in enumerate(f, 1):
        if 'grid' in l and ('state' in l or 'props' in l or 'multiprocessor' in l.lower()):
            print(f"L{idx:4d}: {l.strip()}")
