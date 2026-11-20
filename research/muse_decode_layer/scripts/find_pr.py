with open('submission_s081_keell0renz_890809c.py') as f:
    for i, line in enumerate(f):
        if '"pr"' in line or "'pr'" in line:
            print(f'{i+1}: {line.strip()[:100]}')
