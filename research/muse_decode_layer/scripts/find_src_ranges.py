with open('submission_s081_keell0renz_890809c.py') as f:
    for i, line in enumerate(f):
        if 'ATT7_CUDA_SRC' in line:
            print(f"Line {i+1}: {line.strip()[:100]}")
        if 'MOE_CUDA_SRC' in line:
            print(f"Line {i+1}: {line.strip()[:100]}")
