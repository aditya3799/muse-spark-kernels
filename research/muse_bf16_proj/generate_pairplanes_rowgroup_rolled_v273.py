"""Generate v273: rolled 2/4/8-row-group pair-plane tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_rowgroup_v272.py"
OUTPUT = ROOT / "submission_plain_pairplanes_rowgroup_rolled_v273.py"

text = BASE.read_text(encoding="utf-8")
text = text.replace("v272", "v273").replace("V272", "V273")

old = "_V273_GROUPS = (1, 2, 4)"
new = "_V273_GROUPS = (2, 4, 8)"
if text.count(old) != 1:
    raise RuntimeError("v273 group-list marker mismatch")
text = text.replace(old, new, 1)

old = "#pragma unroll\n  for (int gg = 0; gg < GROUP; ++gg)"
new = "#pragma unroll 1\n  for (int gg = 0; gg < GROUP; ++gg)"
if text.count(old) != 1:
    raise RuntimeError("v273 rolled-loop marker mismatch")
text = text.replace(old, new, 1)

replacements = {
    "  V273_INFO(8192, 512, 4); V273_INFO(1024, 64, 1);":
    "  V273_INFO(8192, 512, 4); V273_INFO(8192, 512, 8);\n"
    "  V273_INFO(1024, 64, 1);",
    "  V273_INFO(1024, 64, 2); V273_INFO(1024, 64, 4);":
    "  V273_INFO(1024, 64, 2); V273_INFO(1024, 64, 4);\n"
    "  V273_INFO(1024, 64, 8);",
    "  V273_INFO(4096, 128, 4);":
    "  V273_INFO(4096, 128, 4); V273_INFO(4096, 128, 8);",
    "  else if (K == 8192 && nth == 512 && group == 4)\n"
    "    cgemv_pair_group_kernel<512,8192,4><<<(npairs+3)/4,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);":
    "  else if (K == 8192 && nth == 512 && group == 4)\n"
    "    cgemv_pair_group_kernel<512,8192,4><<<(npairs+3)/4,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);\n"
    "  else if (K == 8192 && nth == 512 && group == 8)\n"
    "    cgemv_pair_group_kernel<512,8192,8><<<(npairs+7)/8,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);",
    "  else if (K == 1024 && nth == 64 && group == 4)\n"
    "    cgemv_pair_group_kernel<64,1024,4><<<(npairs+3)/4,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);":
    "  else if (K == 1024 && nth == 64 && group == 4)\n"
    "    cgemv_pair_group_kernel<64,1024,4><<<(npairs+3)/4,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);\n"
    "  else if (K == 1024 && nth == 64 && group == 8)\n"
    "    cgemv_pair_group_kernel<64,1024,8><<<(npairs+7)/8,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);",
    "  else if (K == 4096 && nth == 128 && group == 4)\n"
    "    cgemv_pair_group_kernel<128,4096,4><<<(npairs+3)/4,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);":
    "  else if (K == 4096 && nth == 128 && group == 4)\n"
    "    cgemv_pair_group_kernel<128,4096,4><<<(npairs+3)/4,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);\n"
    "  else if (K == 4096 && nth == 128 && group == 8)\n"
    "    cgemv_pair_group_kernel<128,4096,8><<<(npairs+7)/8,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"v273 marker mismatch: {old[:60]!r}")
    text = text.replace(old, new, 1)

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
