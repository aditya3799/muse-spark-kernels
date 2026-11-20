with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py') as f:
    code = f.read()

idx = code.find("ALT_SRCS['v167_helper0']")
end = code.find("ALT_SRCS['v167_main']", idx)
h_code = code[idx:end]
pos = 0
while True:
    p = h_code.find("OKC_WS", pos)
    if p == -1:
        break
    print(repr(h_code[max(0, p-20):p+40]))
    pos = p + 6
