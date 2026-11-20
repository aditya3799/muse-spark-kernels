"""Generate v163: give v162 consumers the full legal 120-register cap."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs256_fourwg_czero_v162.py"
OUT = ROOT / "submission_rs256_fourwg_r120_v163.py"


base = BASE.read_text()
tree = ast.parse(base)
node = next(
    item for item in tree.body
    if isinstance(item, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_V150_CUDA_PAYLOAD" for target in item.targets)
)
cuda = zlib.decompress(base64.b64decode(ast.literal_eval(node.value))).decode()


def replace_once(old: str, new: str) -> None:
    global cuda
    count = cuda.count(old)
    if count != 1:
        raise RuntimeError(f"expected one CUDA match, found {count}: {old[:100]!r}")
    cuda = cuda.replace(old, new, 1)


replace_once(
    "  constexpr int MREG = FOUR ? 112 : (H2 ? 128 : (F2 ? 96 : 232));  // 512*112 + 128*40 = 62464 registers\n",
    "  constexpr int MREG = FOUR ? 120 : (H2 ? 128 : (F2 ? 96 : 232));  // v163: 512*120 + 64*40 = 64000 registers\n",
)
replace_once(
    "  else if constexpr (MREG == 112) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 112;\\n\");\n",
    "  else if constexpr (MREG == 120) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 120;\\n\");\n"
    "  else if constexpr (MREG == 112) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 112;\\n\");\n",
)

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
payload_line = base.splitlines()[node.lineno - 1]
candidate = base.replace(payload_line, f'_V150_CUDA_PAYLOAD = "{new_payload}"', 1)
candidate = candidate.replace("v162", "v163").replace("V162", "V163")
candidate = candidate.replace("regs <= 112", "regs <= 112")

compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
    print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
