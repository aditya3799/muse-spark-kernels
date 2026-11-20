import difflib

with open('submission.py') as f1, open('submission_s112_rq_qkv_l1.py') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines1, lines2, n=3))
print("Total diff lines:", len(diff))
print(''.join(diff))
