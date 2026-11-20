"""Generate v161: three-producer-warp repair of the v159 m=256 diagnostic.

v159's 640-thread kernel needed 62,464 dynamically assigned registers
(512 consumers * 112 + 128 producers * 40), but PTXAS reserved only 61,440.
That made the consumer setmaxnreg handshake impossible; disabling it avoided
the deadlock but serialized every WGMMA.  v161 removes one redundant zeroing
warp, uses 608 threads, and keeps the proven 40-register producer cap.  The
resulting 61,184-register request fits both the architectural file and the
PTXAS reservation while preserving four 112-register consumer warpgroups.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs256_fourwg_v159.py"
OUT = ROOT / "submission_rs256_fourwg3_v161.py"


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
payload = ast.literal_eval(payload_node.value)
cuda = zlib.decompress(base64.b64decode(payload)).decode()


def replace_once(old: str, new: str) -> None:
    global cuda
    count = cuda.count(old)
    if count != 1:
        raise RuntimeError(f"expected one CUDA match, found {count}: {old[:100]!r}")
    cuda = cuda.replace(old, new, 1)


# Three producer warps: W TMA, x TMA, and a single 32-thread zero/walk warp.
# Each zeroing lane simply handles twice the previous number of vector stores.
if cuda.count("(FL & 67108864) ? 640") != 4:
    raise RuntimeError("unexpected v159 FOUR launch-bound count")
cuda = cuda.replace("(FL & 67108864) ? 640", "(FL & 67108864) ? 608")
replace_once(
    "} else if (ZI && warp >= PW + 2) {  // in-kernel zeroing:",
    "} else if (ZI && warp == PW + 2) {  // v161: one-warp in-kernel zeroing:",
)
replace_once("e += 64) {", "e += 32) {")
replace_once('asm volatile("bar.sync 4, 64;\\n" ::: "memory");',
             'asm volatile("bar.sync 4, 32;\\n" ::: "memory");')
replace_once(
    "      if (WK && warp == PW + 3) walk();\n"
    "    } else if (WK && warp == PW + 3) {\n"
    "      walk();\n"
    "    }\n",
    "      if (WK) walk();\n"
    "    }\n",
)

# Restore the dynamic register handshake for FOUR.  At 608 threads PTXAS can
# reserve enough registers for 512*112 + 96*40 = 61,184 registers.
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

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
payload_line = base.splitlines()[payload_node.lineno - 1]
candidate = base.replace(payload_line, f'_V150_CUDA_PAYLOAD = "{new_payload}"', 1)
candidate = candidate.replace("v159", "v161").replace("V159", "V161")
candidate = candidate.replace("four-consumer-WG m256", "four-consumer/three-producer-WG m256")

compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
    print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
