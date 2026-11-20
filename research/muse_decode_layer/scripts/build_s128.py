import os

with open("submission_s127.py", "r", encoding="utf-8") as f:
    src = f.read()

# 1. Update ATT7 unique compilation name for S128
old_att_name = 'name=f"muse_att_v298w1pf_s127_{arch}"'
new_att_name = 'name=f"muse_att_v298w1pf_s128_{arch}"'
assert old_att_name in src, "Failed to find old_att_name"
src = src.replace(old_att_name, new_att_name, 1)

# 2. Update _MA_EXTS with --use_fast_math and -Xptxas -v,-O3
old_ma_tag = 'source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS))).encode()).hexdigest()[:16]'
new_ma_tag = 'source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS, "s128"))).encode()).hexdigest()[:16]'
assert old_ma_tag in src, "Failed to find old_ma_tag"
src = src.replace(old_ma_tag, new_ma_tag, 1)

old_ma_name = 'name=f"glue_moe_{source_tag}_sm{arch}",'
new_ma_name = 'name=f"glue_moe_s128_{source_tag}_sm{arch}",'
assert old_ma_name in src, "Failed to find old_ma_name"
src = src.replace(old_ma_name, new_ma_name, 1)

old_ma_cflags = 'extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",'
new_ma_cflags = 'extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-v,-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",'
assert old_ma_cflags in src, "Failed to find old_ma_cflags"
src = src.replace(old_ma_cflags, new_ma_cflags, 1)

# Ensure strictly Unix LF
src = src.replace("\r\n", "\n").replace("\r", "\n")

out_path = "submission_s128.py"
with open(out_path, "wb") as f:
    f.write(src.encode("ascii"))

sz = os.path.getsize(out_path)
print(f"Built {out_path}: {sz} bytes")
assert sz < 1000000, f"File too large: {sz} bytes >= 1,000,000"
assert b"\r\n" not in open(out_path, "rb").read(), "Found CRLF in output file!"
print("All assertions passed!")
