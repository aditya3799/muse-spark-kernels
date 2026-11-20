import re

with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's search for all #define OKC_WGS or #define OKC_ in MOE_CUDA_SRC
moe_m = re.search(r'MOE_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', text)
if moe_m:
    moe_src = moe_m.group(1)
    defines = re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', moe_src)
    print(f"Total defines in MOE_CUDA_SRC: {len(defines)}")
    for k, v in defines:
        print(f"  {k} = {v.strip()}")
