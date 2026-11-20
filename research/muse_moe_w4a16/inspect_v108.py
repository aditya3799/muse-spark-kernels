with open('submission_v108_mid_l2keep.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'def kernel_fn' in l or 'MID_MAX_M' in l or 'MAIN_DEFINES' in l:
        print(f"L{i}: {l.strip()}")
