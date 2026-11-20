with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
# Find MOE_CUDA_SRC
m = re.search(r"MOE_CUDA_SRC\s*=\s*r?\"\"\"(.*?)\"\"\"", text, re.DOTALL)
if m:
    src = m.group(1)
    knobs = re.findall(r"#ifndef\s+(OKC_[A-Za-z0-9_]+)\s*\n\s*#define\s+\1\s+([^\n]+)", src)
    print(f"Total OKC knobs in MOE_CUDA_SRC: {len(knobs)}")
    for name, val in knobs:
        print(f"  {name:20s} = {val.strip()}")
