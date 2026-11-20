from pathlib import Path


BASE = Path("submission_plain_exact_v180.py")
OUT = Path("submission_plain_qkv1_fixed512_v186.py")


text = BASE.read_text(encoding="utf-8")
old = '''            enable = (valid and cand_us < base_us and
                      core_ratio <= _V105_REQUIRED_CORE_RATIO)'''
new = '''            # v186: the exact QKV m=1 512-thread specialization is a
            # repeated small win whose old 3% prepare gate toggles on timing
            # noise.  Dispatch it deterministically after the existing
            # resource, finiteness, FP64, and one-hot coefficient validation.
            # O and unpooler retain the original conservative 3% gate.
            enable = (valid and (
                layer == 'qkv_gate' or
                (cand_us < base_us and
                 core_ratio <= _V105_REQUIRED_CORE_RATIO)))'''
if text.count(old) != 1:
    raise RuntimeError("unexpected v105 enable anchor count")
text = text.replace(old, new)
text = text.replace("okc_bf16proj_v105_cg_", "okc_bf16proj_v186_cg_", 1)
OUT.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
