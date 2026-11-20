with open('submission_s111_cont_l1_only.py') as f:
    lines = f.readlines()

for i in range(3074, 3095):
    print(f"{i+1}: {lines[i].rstrip()}")
