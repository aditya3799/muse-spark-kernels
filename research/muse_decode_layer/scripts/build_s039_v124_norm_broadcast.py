"""Build S039: S031 plus v124's narrow WGS final-normalization broadcast."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s031_v81_b1.py"
DESTINATION = ROOT / "submission_s039_v124_norm_broadcast.py"
SOURCE_SHA256 = "f5518e5bb7b78321da571d7479aa931db654af1f468ac01d6196cb6d581530ae"
PAYLOAD_PREFIX = 'V99_CUDA_SRC = _zsrc("'
PAYLOAD_SUFFIX = '")\nV99_BW_SRC ='


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S031 hash: {actual_source}")
candidate = source_bytes.decode("utf-8")

payload_start = candidate.index(PAYLOAD_PREFIX) + len(PAYLOAD_PREFIX)
payload_end = candidate.index(PAYLOAD_SUFFIX, payload_start)
payload = candidate[payload_start:payload_end]
cuda = zlib.decompress(base64.b64decode(payload)).decode("utf-8")

old_reduction = """      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
"""
new_reduction = """      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        // This route factor is identical for all 512 threads.  Compute it once
        // per route and broadcast through the existing shared reduction row.
        red[NW * 8 + tid] = qv[tid] >= 0
            ? p.topk_w[tk * 8 + tid] * rsqrtf(tsum * (1.f / HD) + 1e-8f)
            : 0.f;
      }
"""
old_factor = """          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
"""
new_factor = """          const float r = red[NW * 8 + sl];
"""
if cuda.count(old_reduction) != 1 or cuda.count(old_factor) != 1:
    raise RuntimeError(
        f"unexpected embedded v99 normalization sites: "
        f"reduction={cuda.count(old_reduction)}, factor={cuda.count(old_factor)}"
    )
cuda = cuda.replace(old_reduction, new_reduction, 1).replace(old_factor, new_factor, 1)
new_payload = base64.b64encode(zlib.compress(cuda.encode("utf-8"), level=9)).decode("ascii")
candidate = candidate[:payload_start] + new_payload + candidate[payload_end:]

old_name = 'name=f"decode_s022_v99_moe_selmul_{tag}_sm{arch}",'
new_name = 'name=f"decode_s039_v124norm_moe_selmul_{tag}_sm{arch}",'
if candidate.count(old_name) != 1:
    raise RuntimeError("unexpected embedded v99 extension name")
candidate = candidate.replace(old_name, new_name, 1)

required_fragments = (
    "# okc: problem=muse_decode_layer",
    "_e003_attention_kernel_fn = kernel_fn",
    "_V81_CONFIG = {",
    'cfg["batch"] in (1, 16, 32)',
    "decode_s039_v124norm_moe_selmul_",
)
for fragment in required_fragments:
    if fragment not in candidate:
        raise RuntimeError(f"missing S039 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S039 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("embedded v99 CUDA", len(cuda), "bytes")
print("sha256", hashlib.sha256(encoded).hexdigest())
