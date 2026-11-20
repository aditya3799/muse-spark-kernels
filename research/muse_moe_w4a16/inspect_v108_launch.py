with open('submission_v108_mid_l2keep.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'okc_run' in l and 'cudaLaunch' in l:
        print(''.join(lines[i-15:i+15]))
        break
