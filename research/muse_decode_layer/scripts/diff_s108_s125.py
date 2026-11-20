import difflib

with open("submission_s108_att_cgemv.py") as f:
    lines108 = f.readlines()

with open("submission.py") as f:
    lines125 = f.readlines()

diff = list(difflib.unified_diff(lines108, lines125, fromfile="s108", tofile="s125", n=2))
print(f"Total diff lines: {len(diff)}")

for l in diff[:120]:
    print(l, end="")
