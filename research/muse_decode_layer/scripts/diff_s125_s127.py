import difflib

s125 = open('submission_s125.py').readlines()
s127 = open('submission_s127.py').readlines()

diff = list(difflib.unified_diff(s125, s127, fromfile='s125', tofile='s127', n=2))
for line in diff:
    if line.startswith('@@') or line.startswith('---') or line.startswith('+++') or not line.startswith(' '):
        print(line, end='')
