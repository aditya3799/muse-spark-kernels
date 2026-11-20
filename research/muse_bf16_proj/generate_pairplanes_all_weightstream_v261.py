"""Generate v261: L1-no-allocate pair-planes for all m=1 dense routes.

This is the all-route form of v257.  It targets the v130 observation that the
best Pooler policy streamed weights through L1 without an L2 hint and retained
ordinary activation loads.  All four routes compare independently with v221.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_c12_pairplanes_v106.py"
OUTPUT = ROOT / "submission_plain_pairplanes_all_weightstream_v261.py"

text = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
marker = "# v106: exact C12 row-paired code planes for the scored m=1 dense routes."
if donor.count(marker) != 1:
    raise RuntimeError("v106 pair-plane marker mismatch")
block = marker + donor.split(marker, 1)[1]

old_cuda = "_V106_CG_CUDA = r'''\n"
helper = r'''
__device__ __forceinline__ uint4 v261_pair_ld4_na(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
'''
if block.count(old_cuda) != 1:
    raise RuntimeError("v106 pair CUDA marker mismatch")
block = block.replace(old_cuda, old_cuda + helper.strip() + "\n", 1)

for old, new in (
        ("lw[0][j] = __ldg(pv + q);",
         "lw[0][j] = v261_pair_ld4_na(pv + q);"),
        ("lw[1][j] = __ldg(pv + CHUNKS + q);",
         "lw[1][j] = v261_pair_ld4_na(pv + CHUNKS + q);"),
        ("cw[j] = __ldg(pv + 2 * CHUNKS + q);",
         "cw[j] = v261_pair_ld4_na(pv + 2 * CHUNKS + q);")):
    if block.count(old) != 1:
        raise RuntimeError("v106 pair load marker mismatch: " + old)
    block = block.replace(old, new, 1)

block = block.replace("_V106_REQUIRED_CORE_RATIO = 0.97",
                      "_V106_REQUIRED_CORE_RATIO = 0.99", 1)
block = block.replace("okc_bf16proj_v106_c12pair_",
                      "okc_bf16proj_v261_c12pair_", 1)
block = block.replace("v106 c12 pairplanes", "v261 c12 pairplanes")

text += "\n\n\n" + block.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
