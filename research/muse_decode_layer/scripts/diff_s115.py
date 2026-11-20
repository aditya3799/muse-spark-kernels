import difflib

with open('submission.py') as f1, open('submission_s115_cont1_rq_qkv.py') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines1, lines2, n=2))
print("Total diff lines S108 vs S115:", len(diff))
print(''.join(diff))
