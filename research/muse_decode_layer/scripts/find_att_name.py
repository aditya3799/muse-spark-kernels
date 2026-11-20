with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

for line in text.splitlines():
    if 'muse_att_' in line:
        print(line)
