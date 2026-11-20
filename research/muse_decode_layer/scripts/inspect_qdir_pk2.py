import re

with open('submission.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if 'QDIRECT' in line or 'QDIR' in line or 'PK2' in line or 'LSYNC' in line:
        print(f"{i:5d}: {line.rstrip()}")
