"""Build S059: M=32 FT_T=32 with the unused large template compiled out.

S058 proved that the decoder's B4FIN additions push moe_fused<true> over its
register target when FT_T=32 expands JMAX.  The M=32 extension never launches
that BIG template, so S059 makes the extension explicitly small-only while
leaving the executed moe_fused<false> code unchanged.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s052_large_b4_partial_norm.py"
DESTINATION = ROOT / "submission_s059_m32_ft32_smallonly.py"
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
    "#define OKC_FT_T 64\n#ifndef OKC_SPLIT",
    "#ifndef OKC_FT_T\n#define OKC_FT_T 64\n#endif\n#ifndef OKC_SMALLONLY\n#define OKC_SMALLONLY 0\n#endif\n#ifndef OKC_SPLIT",
    "primary compile controls",
)
cuda = replace_once(
    cuda,
    """    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;""",
    """    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0;
#if !OKC_SMALLONLY
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e1;
#endif""",
    "large template attribute reference",
)
cuda = replace_once(
    cuda,
    """    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);""",
    """#if OKC_SMALLONLY
    cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
#else
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
#endif""",
    "PDL large template launch reference",
)
cuda = replace_once(
    cuda,
    """  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);""",
    """#if OKC_SMALLONLY
  okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
#else
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
#endif""",
    "normal large template launch reference",
)
candidate = candidate[:cuda_start] + cuda + candidate[cuda_end:]

flags = '("-DOKC_B4FIN=1", "-DOKC_FT_T=32", "-DOKC_SMALLONLY=1")'
candidate = replace_once(
    candidate,
    '_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))',
    f"_MOE_B4 = _moe_build(tuple(MOE_DEFINES) + {flags})",
    "global M32 dispatch",
)
candidate = replace_once(
    candidate,
    'key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)',
    f'key, defines = "b4_ft32_small", tuple(MOE_DEFINES) + {flags}',
    "state M32 dispatch",
)

required = (
    "#define OKC_SMALLONLY 0",
    "#if OKC_SMALLONLY\n  okc::moe_fused<false>",
    'key, defines = "b4_ft32_small"',
    '"-DOKC_SMALLONLY=1"',
)
for fragment in required:
    if fragment not in candidate:
        raise RuntimeError(f"missing S059 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S059 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
