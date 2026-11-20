import difflib

with open('submission_s090_s081_cgemv_m1.py') as f1, open('submission_s102_rq_cgemv.py') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines1, lines2, n=2))
print("Diff lines count:", len(diff))
print(''.join(diff[:60]))
