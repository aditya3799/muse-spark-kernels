with open('submission_s081_keell0renz_890809c.py') as f:
    for i, line in enumerate(f):
        if 'OKC_WGSUA' in line:
            print(f"Line {i+1}: {line.strip()[:100]}")
