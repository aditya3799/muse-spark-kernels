with open('submission_s111_cont_l1_only.py') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '__launch_bounds__' in l and 'attn_kernel' in l:
        print(f"Line {i+1}: {l.strip()}")
