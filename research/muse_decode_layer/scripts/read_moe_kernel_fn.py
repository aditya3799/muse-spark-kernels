with open('../muse_moe_w4a16/submission.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(4420, min(4470, len(lines))):
    print(f"{i+1}: {lines[i]}", end="")
