with open('submission_s081_keell0renz_890809c.py') as f:
    for i, line in enumerate(f):
        if '__global__' in line and 'moe_fused' in line:
            print(f"Line {i+1}: {line.strip()[:100]}")
