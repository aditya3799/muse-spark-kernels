with open('submission_s081_keell0renz_890809c.py') as f:
    lines = f.readlines()

start = 6115
end = 6175
for i in range(start, end):
    print(f"{i+1:5d}: {lines[i]}", end='')
