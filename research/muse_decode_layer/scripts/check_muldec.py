with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_text = f.read()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_text = f.read()

print("OKC_MULDEC in decode_layer:", "OKC_MULDEC" in dec_text)
print("OKC_MULDEC in muse_moe:", "OKC_MULDEC" in moe_text)

if "OKC_MULDEC" in moe_text:
    import re
    for m in re.finditer(r".{0,50}OKC_MULDEC.{0,50}", moe_text):
        print(" ", m.group(0).strip())
