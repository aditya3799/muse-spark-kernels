"""Build S021: S020 with device-side shard localization fused into v99.

S020 proves that the standalone v99 WGS backend wins for T > 32, but it
materializes ``(topk_ids - local_start) % 256`` before launching the fused MoE.
S021 passes the existing device scalar to the extension and lets v99's native
``id - p.lo`` routing logic consume it inside the persistent kernel.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

# Importing the guarded builder regenerates the known S020 base from protected
# S010.  It also refuses to run if submission.py is no longer the winner hash.
import build_s020_v99_wgs as s020


ROOT = Path(__file__).resolve().parents[1]
SOURCE = s020.DESTINATION
DESTINATION = ROOT / "submission_s021_v99_route_fused.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


candidate = SOURCE.read_text(encoding="utf-8")

# Decode the embedded v99 translation unit, add a pointer to the decoder's
# device-side shard start, and overwrite p.lo before any routing work.
cuda_marker = 'V99_CUDA_SRC = _zsrc("'
cuda_start = candidate.index(cuda_marker) + len(cuda_marker)
cuda_end = candidate.index('")', cuda_start)
cuda = zlib.decompress(base64.b64decode(candidate[cuda_start:cuda_end])).decode("utf-8")

cuda = replace_once(
    cuda,
    "  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;",
    "  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w; const int* route_start;",
    "Params route_start",
)
cuda = replace_once(
    cuda,
    "  const int tid = threadIdx.x;\n  DBG(0);",
    "  const int tid = threadIdx.x;\n  p.lo = __ldg(p.route_start);\n  DBG(0);",
    "device route_start load",
)
cuda = replace_once(
    cuda,
    "void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {",
    "void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor route_start, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {",
    "CUDA okc_run signature",
)
cuda = replace_once(
    cuda,
    "  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();",
    "  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr(); p.route_start = (const int*)route_start.data_ptr();",
    "host Params route_start",
)

payload = base64.b64encode(zlib.compress(cuda.encode("utf-8"), 9)).decode("ascii")
candidate = candidate[:cuda_start] + payload + candidate[cuda_end:]

# Keep the pybind declaration and Python invocation in lock-step with CUDA.
candidate = replace_once(
    candidate,
    "void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);",
    "void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor route_start, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);",
    "CPP okc_run declaration",
)
candidate = replace_once(
    candidate,
    "state[\"w\"], state[\"scratch\"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],",
    "state[\"w\"], state[\"scratch\"], x[m0:m1], ids[m0:m1], data[\"route_start\"], tw[m0:m1], out[m0:m1],",
    "Python okc_run call",
)

# Feed global IDs directly.  v99 already subtracts p.lo and bounds-checks the
# resulting local expert index in both the vector and scalar routing paths.
candidate = replace_once(
    candidate,
    '"topk_ids": (topk_ids - data["local_start"]) % N_EXPERTS,',
    '"topk_ids": topk_ids,\n                "route_start": data["local_start"],',
    "decoder route dispatch",
)
candidate = replace_once(
    candidate,
    "decode_s020_v99_moe_selmul_",
    "decode_s021_v99_moe_selmul_",
    "extension module namespace",
)

encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S021 exceeds the challenge source limit")

DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(candidate):,} bytes)")
print("sha256", hashlib.sha256(DESTINATION.read_bytes()).hexdigest())
