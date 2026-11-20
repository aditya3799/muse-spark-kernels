"""Generate v162: 576-thread four-consumer m256 with consumer-side zeroing.

The 640/608-thread v159/v161 layouts force PTXAS down to 96 registers and
serialize each HGMMA.  v162 keeps only the W and x producer warps, allowing
112 registers for every consumer.  The 512 consumers cooperatively zero their
CTA's output slice while both producer warps start TMA; all 128 split-K CTAs
fit concurrently on H200, so the existing counter rendezvous remains safe.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs256_fourwg_v159.py"
OUT = ROOT / "submission_rs256_fourwg_czero_v162.py"


def assignment(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node
    raise KeyError(name)


base = BASE.read_text()
tree = ast.parse(base)
payload_node = assignment(tree, "_V150_CUDA_PAYLOAD")
cuda = zlib.decompress(base64.b64decode(ast.literal_eval(payload_node.value))).decode()


def replace_once(old: str, new: str) -> None:
    global cuda
    count = cuda.count(old)
    if count != 1:
        raise RuntimeError(f"expected one CUDA match, found {count}: {old[:100]!r}")
    cuda = cuda.replace(old, new, 1)


# Exactly 16 consumer warps plus the two TMA producer warps.  The duplicated
# launch-bound expression is inherited from v159's mechanical host patch.
if cuda.count("(FL & 67108864) ? 640") != 4:
    raise RuntimeError("unexpected v159 FOUR launch-bound count")
cuda = cuda.replace("(FL & 67108864) ? 640", "(FL & 67108864) ? 576")

# With 576 threads the architectural per-thread ceiling is 113 registers, so
# PTXAS can compile the consumer at the requested 112-register dynamic cap.
replace_once(
    "  if constexpr (!FOUR) {\n"
    "    if constexpr (MREG == 232) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\\n\");\n"
    "    else if constexpr (MREG == 128) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 128;\\n\");\n"
    "    else if constexpr (MREG == 112) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 112;\\n\");\n"
    "    else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 96;\\n\");\n"
    "  }\n",
    "  if constexpr (MREG == 232) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\\n\");\n"
    "  else if constexpr (MREG == 128) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 128;\\n\");\n"
    "  else if constexpr (MREG == 112) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 112;\\n\");\n"
    "  else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 96;\\n\");\n",
)

# The old zero/walk warps no longer exist.  Consumers issue disjoint 16-byte
# stores before entering the WGMMA loop.  The two TMA producers run in parallel.
anchor = "  if (SELF || (HALF && warp >= 4)) return;\n"
consumer_zero = r'''  if constexpr (FOUR) {
    if (ZI) {
      const int t = tid;
      const int r0 = (int)((long long)ks * p.m / KS);
      const int r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 512) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      asm volatile("bar.sync 4, 512;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) {
          p.cnt[blockIdx.x] = 0;
          p.cnt[gridDim.x + blockIdx.x] = 0;
        }
      }
    }
  }
'''
replace_once(anchor, consumer_zero + anchor)

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
payload_line = base.splitlines()[payload_node.lineno - 1]
candidate = base.replace(payload_line, f'_V150_CUDA_PAYLOAD = "{new_payload}"', 1)
candidate = candidate.replace("v159", "v162").replace("V159", "V162")
candidate = candidate.replace("four-consumer-WG m256", "four-consumer/consumer-zero m256")

compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
    print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
