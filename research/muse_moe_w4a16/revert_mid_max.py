with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_str = "MID_MAX_M = 32  # v199: test dedicated ring-issuer WSI on M=32"
new_str = "MID_MAX_M = 16  # M 17..63 runs on the WGS MAIN build (v179: faster than the v122c MID kernel at bs32)"

assert old_str in text, "old_str not found!"
text = text.replace(old_str, new_str)

with open('submission.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Reverted MID_MAX_M to 16 in submission.py!")
