with open('submission_s081_keell0renz_890809c.py') as f:
    lines = f.readlines()

for i in range(3640, 3720):
    print(f"{i+1:5d}: {lines[i]}", end='')
