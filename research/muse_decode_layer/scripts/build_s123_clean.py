import os
import re
import py_compile

with open("submission_s122_att_qdir_pk2.py", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8")

# 1. Update _MA_EXTS in submission.py
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

# 2. Set MFAB_LAYER = False
assert "MFAB_LAYER = True" in text, "MFAB_LAYER = True not found"
text = text.replace("MFAB_LAYER = True", "MFAB_LAYER = False", 1)

# Ensure strict LF
text = text.replace("\r\n", "\n")
out_bytes = text.encode("utf-8")

print(f"New file size: {len(out_bytes)} bytes (limit 1,000,000 bytes)")
print(f"Margin: {1000000 - len(out_bytes)} bytes")
assert len(out_bytes) < 1000000, "File too large!"
assert b"\r\n" not in out_bytes, "Found CRLF!"

out_path = "submission.py"
with open(out_path, "wb") as f:
    f.write(out_bytes)

# Verify syntax
py_compile.compile(out_path, doraise=True)
print("Syntax compiled successfully! submission.py is ready.")
