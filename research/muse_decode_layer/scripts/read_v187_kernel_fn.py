text = open(r'd:\OpenKernelChallenge\muse_bf16_proj\submission_plain_qkv1_qkv32_v187.py').read()
idx = text.rfind('def kernel_fn')
print(text[idx:idx+1500])
