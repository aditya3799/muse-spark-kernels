"""Build S058: specialize the decode-layer MoE WGS kernel at M=32 with FT_T=32.

The active S052 checkpoint remains untouched.  Standalone multi-seed testing
showed a small, shape-specific M=32 win from the lower fat-job threshold.  This
candidate changes only the main CUDA source's overridable FT_T definition and
the two M=32 B4FIN dispatch sites; every other token count keeps S052's build.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s052_large_b4_partial_norm.py"
DESTINATION = ROOT / "submission_s058_m32_ft32.py"
SOURCE_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"
CUDA_PREFIX = 'MOE_CUDA_SRC = r"""\n'
CUDA_SUFFIX = '\n"""\nMOE_CPP_SRC = r"""'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S052 hash: {actual_source}")
candidate = source_bytes.decode("utf-8")

# Only the primary MoE CUDA source serves the WGS/B4FIN M=32 branch.  Make its
# default overridable while preserving FT_T=64 for every build without a flag.
cuda_start = candidate.index(CUDA_PREFIX) + len(CUDA_PREFIX)
cuda_end = candidate.index(CUDA_SUFFIX, cuda_start)
cuda = candidate[cuda_start:cuda_end]
cuda = replace_once(
    cuda,
    "#define OKC_FT_T 64\n#ifndef OKC_SPLIT",
    "#ifndef OKC_FT_T\n#define OKC_FT_T 64\n#endif\n#ifndef OKC_SPLIT",
    "primary FT_T default",
)
candidate = candidate[:cuda_start] + cuda + candidate[cuda_end:]

candidate = replace_once(
    candidate,
    '_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))',
    '_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_FT_T=32"))',
    "global M32 dispatch",
)
candidate = replace_once(
    candidate,
    'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)',
    'key, defines = "b4_ft32", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_FT_T=32")',
    "state M32 dispatch",
)

required = (
    "#ifndef OKC_FT_T\n#define OKC_FT_T 64\n#endif",
    '("-DOKC_B4FIN=1", "-DOKC_FT_T=32")',
    'key, defines = "b4_ft32"',
)
for fragment in required:
    if fragment not in candidate:
        raise RuntimeError(f"missing S058 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S058 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
