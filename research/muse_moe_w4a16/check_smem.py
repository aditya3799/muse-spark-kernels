with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py') as f:
    code = f.read()

idx = code.find("ALT_SRCS['v167_main']")
idx2 = code.find("WGS_XS", idx)
print(code[idx2-50:idx2+200])
