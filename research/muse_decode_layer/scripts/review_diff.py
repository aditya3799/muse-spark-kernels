import difflib

with open('submission.py', 'r', encoding='utf-8') as f1, open('submission_s102_rq_cgemv.py', 'r', encoding='utf-8') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = list(difflib.unified_diff(lines1, lines2, fromfile='submission.py (S081)', tofile='submission_s102_rq_cgemv.py', n=2))
print(f"Total diff lines: {len(diff)}")

# Group diff into hunks
hunks = []
cur_hunk = []
for l in diff:
    if l.startswith('@@'):
        if cur_hunk:
            hunks.append(cur_hunk)
        cur_hunk = [l]
    elif cur_hunk:
        cur_hunk.append(l)
if cur_hunk:
    hunks.append(cur_hunk)

print(f"Total hunks: {len(hunks)}")
for idx, h in enumerate(hunks):
    print(f"\n--- HUNK {idx+1} ---")
    for l in h[:20]:
        print(l, end='')
    if len(h) > 20:
        print(f"... ({len(h)-20} more lines in hunk)")
