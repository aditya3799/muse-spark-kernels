import os
import py_compile

with open("submission.py", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8")

# In submission.py (currently has S124), let's revert OKC_FT_T changes
old_ft_t = "#ifndef OKC_FT_T\n#define OKC_FT_T 64\n#endif"
new_ft_t = "#define OKC_FT_T 64"
if old_ft_t in text:
    text = text.replace(old_ft_t, new_ft_t, 1)

old_b4_defs = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0", "-DOKC_FT_T=32")'
new_b4_defs = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0")'
if old_b4_defs in text:
    text = text.replace(old_b4_defs, new_b4_defs)

# Ensure the plan7 optimization is present
old_plan_ft = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)'
new_plan_ft = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target'
if old_plan_ft in text:
    text = text.replace(old_plan_ft, new_plan_ft, 1)

assert new_plan_ft in text, "new_plan_ft not in text!"

# Ensure strict LF
text = text.replace("\r\n", "\n")
out_bytes = text.encode("utf-8")

print(f"New file size: {len(out_bytes)} bytes (limit 1,000,000 bytes)")
print(f"Margin: {1000000 - len(out_bytes)} bytes")
assert len(out_bytes) < 1000000, "File too large!"
assert b"\r\n" not in out_bytes, "Found CRLF!"

out_path = "submission_s125.py"
with open(out_path, "wb") as f:
    f.write(out_bytes)

# Also write directly to submission.py
with open("submission.py", "wb") as f:
    f.write(out_bytes)

# Verify syntax
py_compile.compile("submission.py", doraise=True)
print("Syntax compiled successfully! submission.py is ready for S125.")
