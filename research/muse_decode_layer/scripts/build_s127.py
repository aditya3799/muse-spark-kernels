import os

with open("submission_s126.py", "r", encoding="utf-8") as f:
    src = f.read()

# 1. Update ATT7 unique compilation name for S127
old_att_name = 'name=f"muse_att_v298w1pf_s126_{arch}"'
new_att_name = 'name=f"muse_att_v298w1pf_s127_{arch}"'
assert old_att_name in src, "Failed to find old_att_name"
src = src.replace(old_att_name, new_att_name, 1)

# 2. Add OKC_WPBURST to MoE source at stage wait
old_wgs_wait = 'if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }'
new_wgs_wait = """#ifndef OKC_WPBURST
#define OKC_WPBURST 2
#endif
      if (OKC_WPBURST > 0 && n >= OKC_WPBURST && p.M <= 64) { const int pn = n - OKC_WPBURST; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }"""
assert old_wgs_wait in src, "Failed to find old_wgs_wait"
src = src.replace(old_wgs_wait, new_wgs_wait, 1)

# 3. Add -DOKC_WPBURST=0 to _MOE_B4 definitions (all 4 places)
old_b4_def = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'
new_b4_def = 'tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_WPBURST=0")'
assert src.count(old_b4_def) == 4, f"Expected 4 occurrences of old_b4_def, found {src.count(old_b4_def)}"
src = src.replace(old_b4_def, new_b4_def)

# Ensure strictly Unix LF
src = src.replace("\r\n", "\n").replace("\r", "\n")

out_path = "submission_s127.py"
with open(out_path, "wb") as f:
    f.write(src.encode("ascii"))

sz = os.path.getsize(out_path)
print(f"Built {out_path}: {sz} bytes")
assert sz < 1000000, f"File too large: {sz} bytes >= 1,000,000"
assert b"\r\n" not in open(out_path, "rb").read(), "Found CRLF in output file!"
print("All assertions passed!")
