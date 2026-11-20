text = open(r'd:\OpenKernelChallenge\muse_bf16_proj\submission_plain_qkv1_qkv32_v187.py').read()
idx = text.find('def _v158_prepare')
if idx == -1:
    idx = text.find('vstate')
print(text[idx:idx+2500])
