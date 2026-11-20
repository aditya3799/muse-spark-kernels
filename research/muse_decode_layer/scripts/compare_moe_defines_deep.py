import re

with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_text = f.read()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_text = f.read()

print("--- Decode Layer MoE Defines ---")
for m in re.finditer(r"MOE_DEFINES\s*=\s*\[(.*?)\]", dec_text, re.DOTALL):
    print(m.group(0)[:500])

for m in re.finditer(r"def _main_defines.*?\n\s*return.*?\n", dec_text, re.DOTALL):
    print(m.group(0))

print("\n--- Dedicated MoE Repo Defines ---")
for m in re.finditer(r"MOE_DEFINES\s*=\s*\[(.*?)\]", moe_text, re.DOTALL):
    print(m.group(0)[:500])

for m in re.finditer(r"def _main_defines.*?\n\s*return.*?\n", moe_text, re.DOTALL):
    print(m.group(0))

for m in re.finditer(r"(EXTRA_CFLAGS|cflags|flags)\s*=\s*\[(.*?)\]", moe_text, re.DOTALL):
    print(m.group(0)[:300])
