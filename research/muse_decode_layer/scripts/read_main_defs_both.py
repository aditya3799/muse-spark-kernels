import re

with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    dec_lines = f.readlines()

with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    moe_lines = f.readlines()

print("--- In submission.py ---")
for i, line in enumerate(dec_lines):
    if "def _main_defines" in line:
        for j in range(max(0, i-30), min(len(dec_lines), i+30)):
            print(f"{j+1}: {dec_lines[j]}", end="")
        break

print("\n--- In muse_moe_w4a16/submission.py ---")
for i, line in enumerate(moe_lines):
    if "def _main_defines" in line:
        for j in range(max(0, i-30), min(len(moe_lines), i+30)):
            print(f"{j+1}: {moe_lines[j]}", end="")
        break
