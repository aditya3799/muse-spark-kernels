with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
moe_m = re.search(r'MOE_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', text)
moe_src = moe_m.group(1)

for knob in ['OKC_GATHB', 'OKC_WGSPR', 'OKC_WGSDFIN', 'OKC_WGSFCI', 'OKC_WGSGS', 'OKC_WGSU', 'OKC_WGSUB', 'OKC_WGSNX', 'OKC_WGSREORD']:
    matches = [line.strip() for line in moe_src.splitlines() if knob in line and not line.strip().startswith('//')]
    print(f"\nKnob: {knob}")
    for m in matches[:5]:
        print(f"  {m}")
