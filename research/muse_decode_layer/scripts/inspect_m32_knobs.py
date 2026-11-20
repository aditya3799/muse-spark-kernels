with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re
for d in ["OKC_FT_T", "OKC_MMAX", "OKC_NO_BIG", "OKC_WGSNS", "OKC_RLOAD"]:
    matches = re.findall(rf".{{0,30}}{d}.{{0,40}}", text)
    print(f"=== {d} ({len(matches)} occurrences) ===")
    for m in matches[:5]:
        print(" ", m.strip())
