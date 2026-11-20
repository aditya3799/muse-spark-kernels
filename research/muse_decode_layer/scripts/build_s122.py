import re

with open('submission.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update attention compilation flags: enable OKC_QDIRECT=1 and OKC_PK2=1
old_att_line = '_ATT7 = load_inline(name=f"muse_att_v298w1pf_s120_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,\n                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],'
new_att_line = '_ATT7 = load_inline(name=f"muse_att_v298w1pf_s122_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,\n                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=1", "-DOKC_PK2=1", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],'

assert content.count(old_att_line) == 1, f"old_att_line count: {content.count(old_att_line)}"
content = content.replace(old_att_line, new_att_line, 1)

# Verify size and line endings
encoded = content.encode('utf-8').replace(b'\r\n', b'\n')
print(f"Total size: {len(encoded)} bytes (limit: 1,000,000, margin: {1000000 - len(encoded)})")
assert len(encoded) < 1000000, "File too large!"
assert b'\r' not in encoded, "CRLF found!"

with open('submission_s122_att_qdir_pk2.py', 'wb') as f:
    f.write(encoded)

print("Saved submission_s122_att_qdir_pk2.py successfully.")
