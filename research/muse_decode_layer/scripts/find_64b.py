with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

for line in text.splitlines():
    if '64B' in line:
        print(line[:100])
