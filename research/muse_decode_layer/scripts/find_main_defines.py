with open('submission.py') as f:
    text = f.read()

for line in text.splitlines():
    if '_main_defines' in line or 'MOE_DEFINES =' in line:
        print(line)
