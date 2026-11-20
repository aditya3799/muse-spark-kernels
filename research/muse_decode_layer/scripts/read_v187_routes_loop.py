text = open(r'd:\OpenKernelChallenge\muse_bf16_proj\submission_plain_qkv1_qkv32_v187.py').read()
idx = text.find('for (m, layer), variant in _V158_ROUTES.items():')
if idx == -1:
    idx = text.find('_V158_ROUTES')
    idx = text.find('for ', idx)
print(text[idx:idx+2500])
