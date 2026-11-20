import os
import re
import py_compile

with open("submission.py", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8")

# 1. Wrap OKC_FT_T with #ifndef in MOE_CUDA_SRC
old_ft_t = "#define OKC_FT_T 64"
new_ft_t = "#ifndef OKC_FT_T\n#define OKC_FT_T 64\n#endif"

assert old_ft_t in text, "old_ft_t not found"
text = text.replace(old_ft_t, new_ft_t, 1)

# 2. Add -DOKC_FT_T=32 to _MOE_B4 defines
old_b4_defs = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0")'
new_b4_defs = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0", "-DOKC_FT_T=32")'

assert old_b4_defs in text, "old_b4_defs not found"
text = text.replace(old_b4_defs, new_b4_defs)

# 3. In _plan7, use target (264 CTAs) instead of (3 * sms) for L == 1 full layer
old_plan_ft = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)'
new_plan_ft = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target'

assert old_plan_ft in text, "old_plan_ft not found"
text = text.replace(old_plan_ft, new_plan_ft, 1)

# Ensure strict LF
text = text.replace("\r\n", "\n")
out_bytes = text.encode("utf-8")

print(f"New file size: {len(out_bytes)} bytes (limit 1,000,000 bytes)")
print(f"Margin: {1000000 - len(out_bytes)} bytes")
assert len(out_bytes) < 1000000, "File too large!"
assert b"\r\n" not in out_bytes, "Found CRLF!"

out_path = "submission_s124.py"
with open(out_path, "wb") as f:
    f.write(out_bytes)

# Verify syntax
py_compile.compile(out_path, doraise=True)
print("Syntax compiled successfully! submission_s124.py is ready.")
