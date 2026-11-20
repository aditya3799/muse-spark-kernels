with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v178_hybrid_m32_retry.py') as f:
    text = f.read()

idx = text.find('OKC_WGSUB')
if idx != -1:
    print(text[idx-200:idx+300])
else:
    print("OKC_WGSUB not found in v178")
