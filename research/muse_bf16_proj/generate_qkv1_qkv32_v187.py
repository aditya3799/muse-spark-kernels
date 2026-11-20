from pathlib import Path


BASE = Path("submission_plain_qkv1_fixed512_v186.py")
V158 = Path("submission_plain_composite_v182.py")
OUT = Path("submission_plain_qkv1_qkv32_v187.py")


text = BASE.read_text(encoding="utf-8")

# Add only the already validated RS32 QKV specialization to the faithful-RS128
# extension.  The extension name is source-hashed, so the new case gets its own
# build artifact without changing any incumbent case.
old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v150 switch anchor count")
text = text.replace(old_switch, new_switch)

# Reuse the readable, plain-source v158 validation/tournament wrapper from v182.
# v182's losing QKV/m256 cuBLASLt seed lives earlier in that file and is
# intentionally not copied here.
v158_text = V158.read_text(encoding="utf-8")
marker = "# ---------------------------------------------------------------- v158: exact RS16/RS32 guarded tournament."
if v158_text.count(marker) != 1:
    raise RuntimeError("unexpected v158 block marker count")
v158_block = v158_text[v158_text.index(marker):]
text = text.rstrip() + "\n\n\n" + v158_block.rstrip() + "\n"

with OUT.open("w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
