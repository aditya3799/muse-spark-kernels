with open('submission_s081_keell0renz_890809c.py') as f:
    lines = f.readlines()

for i in range(3180, 3260):
    print(f"{i+1:4d}: {lines[i]}", end='')
