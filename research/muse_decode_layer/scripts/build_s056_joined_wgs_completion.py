"""Build S056: join WGS weight and activation completion barriers.

The active S052 checkpoint remains untouched. This candidate ports the
standalone WGS scheduler's joined-completion protocol only into the main MoE
CUDA source used by M=32/128/256. The small-token alternate source, routing,
math, and finalizers are unchanged.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s052_large_b4_partial_norm.py"
DESTINATION = ROOT / "submission_s056_joined_wgs_completion.py"
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

cuda_start = candidate.index(CUDA_PREFIX) + len(CUDA_PREFIX)
cuda_end = candidate.index(CUDA_SUFFIX, cuda_start)
cuda = candidate[cuda_start:cuda_end]

cuda = replace_once(
    cuda,
    """constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)\nconstexpr int WGS_NT = FT_T / 8;""",
    """constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)\nstatic_assert(WGS_NS == WGS_NX, "Joined completion requires equal ring depths");\nconstexpr int WGS_NT = FT_T / 8;""",
    "equal ring-depth guard",
)
cuda = replace_once(
    cuda,
    """      uint64_t* bar = S.bars + BX_F + xs;""",
    """      // Join this stage's activation and weight transactions. The consumer's\n      // single acquire below observes both completions.\n      uint64_t* bar = S.bars + BW_F + xs;""",
    "activation completion barrier",
)
cuda = replace_once(
    cuda,
    """    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);""",
    """    // The joined acquire covers both the weight and activation transactions.\n    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n    const unsigned char* pc = pc0 + ws * UNIT_BYTES;""",
    "consumer joined acquire",
)
cuda = replace_once(
    cuda,
    """    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);""",
    """    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 2 : k < BX_F ? NW : k < BX_E ? 1 : NW);""",
    "joined full-barrier arrival count",
)

candidate = candidate[:cuda_start] + cuda + candidate[cuda_end:]

required = (
    'static_assert(WGS_NS == WGS_NX, "Joined completion requires equal ring depths");',
    "uint64_t* bar = S.bars + BW_F + xs;",
    "k < BW_E ? 2 : k < BX_F ? NW",
)
for fragment in required:
    if fragment not in candidate:
        raise RuntimeError(f"missing S056 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S056 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
