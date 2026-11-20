import re

with open('submission.py', 'r', encoding='utf-8') as f:
    s = f.read()

m = re.search(r'ATT7_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', s)
att_src = m.group(1)

# Let's check which defines are present in att_src
defines = re.findall(r'#define\s+(OKC_\w+)\s+([^\n/]+)', att_src)
print("Defines in ATT7_CUDA_SRC:")
for k, v in defines:
    print(f"  {k} = {v.strip()}")
