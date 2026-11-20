"""Generate v154: two-k64-stage RS128 consumer groups for pooler m=128.

This reuses v153's standalone validation/bake wrapper but changes its two
candidate cases to GS=2: WM0 batches eight WGMMA operations per commit, while
WM1 also decodes the following two-stage group beneath the current group.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150
import generate_rs128_wm_v153 as v153


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_gs2_v154.py"

gs_case = (
    v150.case
    + "\n    case 1: return rq_go<128, 5, 2, 0, 0, 0, 42999939, 1>"
      "(p, x, out, (int)KS, (int)info);"
    + "\n    case 2: return rq_go<128, 5, 2, 0, 1, 0, 42999939, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
if v150.cuda_source.count(v150.case) != 1:
    raise RuntimeError("v150 case anchor changed")
gs_cuda = v150.cuda_source.replace(v150.case, gs_case)
gs_payload = base64.b64encode(zlib.compress(gs_cuda.encode(), 9)).decode()

# v153's appended Python is deliberately generic over cases 1/2.  Rename it
# and replace only the embedded CUDA payload/hash and human-facing labels.
append = v153.append
append = append.replace(v153.wm_payload, gs_payload)
append = append.replace(
    hashlib.sha256(v153.wm_cuda.encode()).hexdigest().upper(),
    hashlib.sha256(gs_cuda.encode()).hexdigest().upper(),
)
append = append.replace("V153", "V154").replace("v153", "v154")
append = append.replace(
    "overlap C12 decode with WGMMA for pooler m=128",
    "batch two C12 k64 stages per consumer group for pooler m=128",
)
append = append.replace("rs128_wm_", "rs128_gs2_")
append = append.replace("WM%d", "GS2 variant %d")
append = append.replace("WM unavailable", "GS2 unavailable")
append = append.replace("exact pooler WM", "exact pooler GS2 variant")
append = append.replace("pooler WM", "pooler GS2 variant")
append = append.replace("v150_WM0", "v150_GS1_WM0")
append = append.replace("pooler picked WM", "pooler picked GS2 variant")

base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("gs2_cuda_sha256", hashlib.sha256(gs_cuda.encode()).hexdigest().upper())
print("bytes", len(candidate.encode()))
