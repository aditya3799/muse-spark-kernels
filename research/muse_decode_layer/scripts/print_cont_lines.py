with open('submission_s111_cont_l1_only.py') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'bool CONT' in l:
        print(f"Line {i+1}: {l.strip()}")
        for j in range(max(0, i-5), min(len(lines), i+6)):
            print(f"  {j+1}: {lines[j].rstrip()}")
