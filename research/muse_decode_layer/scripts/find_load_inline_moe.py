with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
for m in re.finditer(r"load_inline\((.*?)\)", text, re.DOTALL):
    print(m.group(0)[:300])
    print("---")
