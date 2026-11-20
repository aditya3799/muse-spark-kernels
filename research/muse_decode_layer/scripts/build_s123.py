import re
import os

with open("submission.py", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8")

# 1. Update _MA_EXTS in submission.py
# Check line 6940:
old_ma_ext_block = """        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        import hashlib as _native_build_hash
        source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS))).encode()).hexdigest()[:16]
        _MA_EXTS[defines] = load_inline(
            name=f"glue_moe_{source_tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )"""

new_ma_ext_block = """        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        import hashlib as _native_build_hash
        source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS, "s123"))).encode()).hexdigest()[:16]
        _MA_EXTS[defines] = load_inline(
            name=f"glue_moe_s123_{source_tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-v,-O3",
                               f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )"""

assert old_ma_ext_block in text, "old_ma_ext_block not found in text"
text = text.replace(old_ma_ext_block, new_ma_ext_block, 1)

# 2. Disable MFAB_LAYER:
# Replace the MFAB_LAYER diagnostics block with simple prepare pass-through
old_mfab_pattern = r"# -+ MFIN: paired in-prepare timing.*?\ndef prepare\(weights\):.*?\n    return st\n"
m = re.search(old_mfab_pattern, text, re.DOTALL)
assert m is not None, "MFAB block not found"
print("Found MFAB block length:", len(m.group(0)))

new_mfab_replacement = """# MFAB_LAYER disabled for production speed
prepare = _mfin_prior_prepare
"""

text = text[:m.start()] + new_mfab_replacement + text[m.end():]

# Ensure strict LF
text = text.replace("\r\n", "\n")
out_bytes = text.encode("utf-8")

print(f"New file size: {len(out_bytes)} bytes (limit 1,000,000 bytes)")
print(f"Margin: {1000000 - len(out_bytes)} bytes")
assert len(out_bytes) < 1000000, "File too large!"
assert b"\r\n" not in out_bytes, "Found CRLF!"

with open("submission_s123.py", "wb") as f:
    f.write(out_bytes)

print("Wrote submission_s123.py successfully!")
