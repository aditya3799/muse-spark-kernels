import re

with open('submission.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. wpburst patch in MOE_CUDA_SRC
old_wp = "if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }"
new_wp = "#ifndef OKC_WPBURST\n#define OKC_WPBURST 2\n#endif\n      if (OKC_WPBURST > 0 && n >= OKC_WPBURST && p.M <= 64) { const int pn = n - OKC_WPBURST; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }"
assert content.count(old_wp) == 1, f"old_wp count: {content.count(old_wp)}"
content = content.replace(old_wp, new_wp, 1)

# 2. Add -DOKC_WPBURST=0 to MOE_B4
old_b4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'
new_b4 = 'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0")'
assert content.count(old_b4) == 1, f"old_b4 count: {content.count(old_b4)}"
content = content.replace(old_b4, new_b4, 1)

old_b4_prep = '[(32, tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)), (128, main), (256, main)]'
new_b4_prep = '[(32, tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0")), (128, main), (256, main)]'
assert content.count(old_b4_prep) == 1, f"old_b4_prep count: {content.count(old_b4_prep)}"
content = content.replace(old_b4_prep, new_b4_prep, 1)

old_b4_build = '_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))'
new_b4_build = '_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0"))'
assert content.count(old_b4_build) == 1, f"old_b4_build count: {content.count(old_b4_build)}"
content = content.replace(old_b4_build, new_b4_build, 1)

# 3. MOE_MID_DEFINES with t2_wsi and WSBLK=1
# In line 13384:
old_t2 = 'MOE_TINY2_DEFINES = tuple(d for d in MOE_TINY2_DEFINES if not d.startswith("@")) + ("@t2_wsi",)'
new_t2 = """MOE_TINY2_DEFINES = tuple(d for d in MOE_TINY2_DEFINES if not d.startswith("@")) + ("@t2_wsi",)
MOE_MID_DEFINES = tuple(d for d in MOE_MID_DEFINES if not d.startswith("@") and not d.startswith("-DOKC_WSBLK")) + ("-DOKC_WSBLK=1", "-DOKC_DF=0", "@t2_wsi")"""
assert content.count(old_t2) == 1, f"old_t2 count: {content.count(old_t2)}"
content = content.replace(old_t2, new_t2, 1)

# 4. Remove _df_patch to save space and avoid unused code
df_start = content.find('def _df_patch(c):')
df_end = content.find('MOE_MID_DEFINES = tuple("@moe_helper0df"', df_start)
df_end_line = content.find('\n', df_end) + 1
print(f"Removing _df_patch block from {df_start} to {df_end_line} ({df_end_line - df_start} bytes)")
content = content[:df_start] + content[df_end_line:]

# Verify byte count and LF
encoded = content.encode('utf-8')
print(f"Total size: {len(encoded)} bytes (limit: 1,000,000, remaining: {1000000 - len(encoded)})")
assert len(encoded) < 1000000, "File too large!"
assert b'\r' not in encoded, "CRLF found!"

with open('submission_s121_mid_wsi_wpburst.py', 'wb') as f:
    f.write(encoded)

print("Saved submission_s121_mid_wsi_wpburst.py successfully.")
