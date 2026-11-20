text = open(r'd:\OpenKernelChallenge\muse_bf16_proj\submission_plain_composite_v182.py').read()
idx = text.find('# ---------------------------------------------------------------- v158: exact RS16/RS32 guarded tournament.')
print(text[idx:idx+2500])
