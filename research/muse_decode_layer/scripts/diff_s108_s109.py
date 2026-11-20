import difflib

with open('submission_s108_att_cgemv.py') as f1, open('submission_s109_cont1.py') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines1, lines2, n=3))
print("Total diff lines S108 vs S109:", len(diff))
print(''.join(diff))
