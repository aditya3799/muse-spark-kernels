import difflib

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py') as f1, open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v172_hybrid_small.py') as f2:
    s1 = f1.readlines()
    s2 = f2.readlines()

print(f"submission.py lines: {len(s1)}, v172 lines: {len(s2)}")
diff = list(difflib.unified_diff(s1, s2, n=2))
print(f"Diff lines count: {len(diff)}")
for line in diff[:60]:
    print(line, end='')
