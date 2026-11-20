import re

with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_text = f.read()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_text = f.read()

# Let's inspect MOE_DEFINES / MAIN_DEFINES
print("=== submission.py MOE defines ===")
for m in re.finditer(r"MOE_.*DEFINES\s*=.*", dec_text):
    print(m.group(0))

print("\n=== muse_moe_w4a16 MAIN/TINY defines ===")
for m in re.finditer(r"(?:MAIN|TINY|TINY2|MID)_DEFINES\s*=.*", moe_text):
    print(m.group(0))

print("\n=== Dispatch rules ===")
for m in re.finditer(r"def _(ma_pick|pick_build|dispatch|ma_choose).*", dec_text):
    print(m.group(0))
for m in re.finditer(r"def _(ma_pick|pick_build|dispatch|ma_choose).*", moe_text):
    print(m.group(0))
