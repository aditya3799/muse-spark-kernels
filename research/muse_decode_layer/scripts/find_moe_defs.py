import re

with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_lines = f.readlines()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_lines = f.readlines()

print("--- Decode Layer: Searching for MOE_DEFINES ---")
for i, l in enumerate(dec_lines):
    if re.match(r"\s*(MOE_DEFINES|MOE_TINY_DEFINES|MOE_TINY2_DEFINES|MOE_MID_DEFINES)\s*=", l):
        print(f"Line {i+1}: {l.strip()}")
        for k in range(i+1, min(i+15, len(dec_lines))):
            print(f"  {dec_lines[k].strip()}")
            if ")" in dec_lines[k] or "]" in dec_lines[k]:
                break

print("\n--- MoE dedicated repo: Searching for MAIN_DEFINES / TINY_DEFINES ---")
for i, l in enumerate(moe_lines):
    if re.match(r"\s*(MAIN_DEFINES|TINY_DEFINES|TINY2_DEFINES|MID_DEFINES)\s*=", l):
        print(f"Line {i+1}: {l.strip()}")
        for k in range(i+1, min(i+15, len(moe_lines))):
            print(f"  {moe_lines[k].strip()}")
            if ")" in moe_lines[k] or "]" in moe_lines[k]:
                break
