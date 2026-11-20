with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission_v178_hybrid_m32_retry.py') as f:
    text = f.read()

idx = text.find('#define OKC_WGSU 80')
if idx != -1:
    print(text[idx-500:idx+500])
