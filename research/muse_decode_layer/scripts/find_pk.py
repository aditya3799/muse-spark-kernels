with open('submission_s081_keell0renz_890809c.py') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'PK =' in line or 'constexpr int PK' in line or 'PK' in line and 'TILE' in line:
        print(f"Line {i+1}: {line.strip()[:100]}")
