with open('submission_s081_keell0renz_890809c.py') as f:
    for i, line in enumerate(f):
        if 'def bfp_fused_pooler_router' in line:
            print(f'{i+1}: {line.strip()[:100]}')
