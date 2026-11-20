import re

with open(r'd:\OpenKernelChallenge\muse_attention_decode\submission.py', encoding='utf-8') as f:
    attn_py = f.read()

with open('submission.py', encoding='utf-8') as f:
    layer_py = f.read()

def get_cuda_src(src, var_name):
    pos = src.find(f'{var_name} = r"""')
    if pos == -1: pos = src.find(f"{var_name} = r'''")
    if pos == -1: pos = src.find(f'{var_name} = """')
    start = src.find('\n', pos) + 1
    # find end of string
    quote = src[pos + len(var_name) + 3 : pos + len(var_name) + 6]
    end = src.find(quote, start)
    return src[start:end]

attn_cuda = get_cuda_src(attn_py, 'CUDA_SRC')
layer_cuda = get_cuda_src(layer_py, 'ATT7_CUDA_SRC')

print('attn_cuda lines:', len(attn_cuda.splitlines()), 'bytes:', len(attn_cuda))
print('layer_cuda lines:', len(layer_cuda.splitlines()), 'bytes:', len(layer_cuda))

# check difference in #defines or key functions
attn_defines = [l.strip() for l in attn_cuda.splitlines() if l.strip().startswith('#define OKC_')]
layer_defines = [l.strip() for l in layer_cuda.splitlines() if l.strip().startswith('#define OKC_')]

print('\n=== Defines in attn_cuda but not layer_cuda ===')
for d in attn_defines:
    if d not in layer_defines:
        print(' ', d)

print('\n=== Defines in layer_cuda but not attn_cuda ===')
for d in layer_defines:
    if d not in attn_defines:
        print(' ', d)
