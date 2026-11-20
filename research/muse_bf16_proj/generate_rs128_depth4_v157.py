"""Generate v157: RS128 cursor/short-apply with a four-stage ring.

The protected v150 candidate uses the upstream-proven depth-five instance.
Upstream's H200 sweep found the otherwise identical pre-cursor depth-four
instance faster, but never combined depth four with the later cursor and
short-apply improvements.  This generator changes only the template ring
depth and retains v150's exact packer, validation, route gates, and v105
fallback.
"""

from __future__ import annotations

import base64
import hashlib
import zlib

import generate_rs128_faithful_v150 as v150


OUT = v150.ROOT / "submission_rs128_depth4_v157.py"

# Name is diagnostic only.  The CUDA dispatch is instantiated directly from
# the tuple fields, so this legitimately tests a combination absent from the
# upstream RQV table.
variant = ("d2s4hzwP", 128, 4, 1, 0, 0, 0, 42999939)
case = "    case 0: return rq_go<128, 4, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);"

cuda_source = v150.rqcu.replace(
    "@@WGMMA@@",
    v150.wg_namespace["_rqwg"](16)
    + v150.wg_namespace["_rqwg"](32)
    + v150.wg_namespace["_rqwg"](128)
    + v150.wg_namespace["_rqwg"](256)
    + v150.wg_namespace["_rqss"](128)
    + v150.wg_namespace["_rqss"](256),
).replace("@@CASES@@", case)
cuda_source = cuda_source.replace(
    "mbarrier.arrive.relaxed.cta.shared::cta.b64",
    "mbarrier.arrive.shared::cta.b64",
)
cuda_payload = base64.b64encode(zlib.compress(cuda_source.encode(), 9)).decode()

prelude = v150.prelude.replace(v150.cuda_payload, cuda_payload).replace(
    repr(v150.variant), repr(variant)
)
candidate = (
    v150.base
    + prelude
    + "\n\n"
    + v150.pack_source
    + "\n\n"
    + v150.integration.lstrip()
)
compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("variant", variant)
    print("cuda_sha256", hashlib.sha256(cuda_source.encode()).hexdigest().upper())
    print("baseline_prefix", candidate.encode()[: len(v150.base_bytes)] == v150.base_bytes)
    print("bytes", len(candidate.encode()))
