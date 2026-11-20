with open('submission_v186_hybrid_m32_m128.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'def kernel_fn' in line:
        print(''.join(lines[i:i+40]))
        break
