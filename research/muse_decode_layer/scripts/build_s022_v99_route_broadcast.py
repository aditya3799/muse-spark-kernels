"""Build S022: S021 with one device shard-start load per CTA.

S021 removed the standalone ID-localization launch, but every CUDA thread loaded
the same device scalar.  S022 has thread 0 publish it through the already
allocated, not-yet-used routing-count shared memory before helper warps split.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import build_s021_v99_route_fused as s021


ROOT = Path(__file__).resolve().parents[1]
SOURCE = s021.DESTINATION
DESTINATION = ROOT / "submission_s022_v99_route_broadcast.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


candidate = SOURCE.read_text(encoding="utf-8")
marker = 'V99_CUDA_SRC = _zsrc("'
start = candidate.index(marker) + len(marker)
end = candidate.index('")', start)
cuda = zlib.decompress(base64.b64decode(candidate[start:end])).decode("utf-8")

cuda = replace_once(
    cuda,
    "  p.lo = __ldg(p.route_start);",
    "  if (tid == 0) S.fcnt[0] = __ldg(p.route_start);\n"
    "  __syncthreads();\n"
    "  p.lo = *reinterpret_cast<volatile int*>(S.fcnt);",
    "CTA route-start broadcast",
)

payload = base64.b64encode(zlib.compress(cuda.encode("utf-8"), 9)).decode("ascii")
candidate = candidate[:start] + payload + candidate[end:]
candidate = replace_once(
    candidate,
    "decode_s021_v99_moe_selmul_",
    "decode_s022_v99_moe_selmul_",
    "extension module namespace",
)

encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S022 exceeds the challenge source limit")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
