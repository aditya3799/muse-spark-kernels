with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_text = f.read()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_text = f.read()

import re

# Find MOE_CUDA_SRC in dec_text
m_dec = re.search(r"MOE_CUDA_SRC\s*=\s*r?\"\"\"(.*?)\"\"\"", dec_text, re.DOTALL)
m_moe = re.search(r"(?:MOE_)?CUDA_SRC\s*=\s*r?\"\"\"(.*?)\"\"\"", moe_text, re.DOTALL)

print("Dec CUDA SRC found:", bool(m_dec))
print("MoE CUDA SRC found:", bool(m_moe))

if m_dec and m_moe:
    src_dec = m_dec.group(1)
    src_moe = m_moe.group(1)
    print(f"Dec CUDA SRC len: {len(src_dec)}")
    print(f"MoE CUDA SRC len: {len(src_moe)}")
    print(f"Are they identical? {src_dec == src_moe}")
    if src_dec != src_moe:
        import difflib
        diff = list(difflib.unified_diff(
            src_dec.splitlines(keepends=True),
            src_moe.splitlines(keepends=True),
            fromfile="decode_layer",
            tofile="muse_moe",
            n=3
        ))
        print("Diff lines:", len(diff))
        for l in diff[:50]:
            print(l, end="")
