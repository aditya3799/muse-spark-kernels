with open('submission_s102_rq_cgemv.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'def _rqbake(' in l:
        for j in range(i, min(i + 70, len(lines))):
            print(f"{j+1}: {lines[j]}", end='')
        break
