import difflib

with open(r'd:\OpenKernelChallenge\muse_attention_decode\submission.py', encoding='utf-8') as f:
    attn_full = f.read()

with open('submission.py', encoding='utf-8') as f:
    layer_full = f.read()

# Find compile flags for attention
def find_flags(src):
    flags = []
    for line in src.splitlines():
        if 'load_inline' in line or 'extra_cuda_cflags' in line or 'muse_att_' in line or 'okc_attn_' in line:
            flags.append(line.strip())
    return flags

print('=== Flags in muse_attention_decode ===')
for l in find_flags(attn_full):
    print(' ', l)

print('=== Flags in muse_decode_layer ===')
for l in find_flags(layer_full):
    print(' ', l)
