with open('submission.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if l.startswith('TINY_DEFINES') or l.startswith('TINY2_DEFINES') or l.startswith('MID_DEFINES'):
        print(f"L{i}: {l.strip()}")
