with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v172_hybrid_small.py') as f:
    lines = f.readlines()

for i in range(4175, 4210):
    print(lines[i], end='')
