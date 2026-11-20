"""Generate v164: pre-materialize all four WGMMA descriptors for v163."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs256_fourwg_r120_v163.py"
OUT = ROOT / "submission_rs256_fourwg_desc4_v164.py"

base = BASE.read_text()
tree = ast.parse(base)
node = next(
    item for item in tree.body
    if isinstance(item, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_V150_CUDA_PAYLOAD" for target in item.targets)
)
cuda = zlib.decompress(base64.b64decode(ast.literal_eval(node.value))).decode()

start = cuda.index("__device__ __forceinline__ void rq_wgmma4_128")
end = cuda.index("template <> __device__ __forceinline__ void rq_wgmma<256>", start)
old_helper = cuda[start:end]

dregs = ", ".join(f"%{i}" for i in range(64))
aregs = [", ".join(f"%{64 + 4*q + i}" for i in range(4)) for q in range(4)]
mma = ""
for q in range(4):
    mma += (
        f"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        f"{{{dregs}}}, {{{aregs[q]}}}, %{80 + q}, p{0 if q == 0 else 1}, 1, 1, 0;\\n"
    )
outputs = ", ".join(f'\"+f\"(D[{i}])' for i in range(64))
inputs = ", ".join([
    *(f'\"r\"(a[{i}])' for i in range(16)),
    '\"l\"(d0)', '\"l\"(d1)', '\"l\"(d2)', '\"l\"(d3)', '\"r\"(scd)',
])
new_helper = (
    "__device__ __forceinline__ void rq_wgmma4_128(float (&D)[64], const uint32_t* a, "
    "uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3, int scd) {\n"
    "  asm volatile(\"{\\n.reg .pred p0, p1;\\n"
    "setp.ne.b32 p0, %84, 0;\\nsetp.eq.b32 p1, %84, %84;\\n"
    + mma
    + "}\\n\"\n    : " + outputs + "\n    : " + inputs + ");\n}\n"
)
cuda = cuda[:start] + new_helper + cuda[end:]

call = "rq_wgmma4_128(D, A, d, s != 0);"
if cuda.count(call) != 1:
    raise RuntimeError("unexpected fused-helper call count")
cuda = cuda.replace(call, "rq_wgmma4_128(D, A, d, d + 2, d + 4, d + 6, s != 0);", 1)

new_payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
payload_line = base.splitlines()[node.lineno - 1]
candidate = base.replace(payload_line, f'_V150_CUDA_PAYLOAD = "{new_payload}"', 1)
candidate = candidate.replace("v163", "v164").replace("V163", "V164")

compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("old_helper_sha256", hashlib.sha256(old_helper.encode()).hexdigest().upper())
    print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
    print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
