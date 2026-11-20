with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py') as f:
    code = f.read()

idx = code.find("ALT_SRCS['v167_main']")
m_code = code[idx:]

pos = 0
while True:
    p = m_code.find("__nanosleep(32)", pos)
    if p == -1:
        break
    print(f"Match at {p}:")
    print(repr(m_code[max(0, p-60):p+30]))
    pos = p + 15
