import re

with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_text = f.read()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_text = f.read()

def find_defs(text, name):
    print(f"=== {name} ===")
    matches = re.findall(r"(?:MOE_DEFINES|MAIN_DEFINES|MOE_FLAGS|MOE_CFLAGS|BASE_DEFINES)\s*=\s*\[(.*?)\]", text, re.DOTALL)
    for m in matches:
        defs = [x.strip().strip('"').strip("'") for x in m.split(",") if x.strip()]
        print(f"Count: {len(defs)}")
        print("\n".join("  " + d for d in defs[:30]))
        if len(defs) > 30:
            print("  ... and", len(defs)-30, "more")

find_defs(dec_text, "Decode Layer")
find_defs(moe_text, "Dedicated MoE Repo")
