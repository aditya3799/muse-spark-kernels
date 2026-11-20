"""Build S060: complete the M=32 FT32 port with the standalone switch guards.

S059 compiled impossible WGSMMA<5..8> cases even though FT_T=32 limits NTT to
four.  The standalone implementation guards those instantiations with
OKC_FT_T; this builder adds the same guard to the otherwise isolated S059.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s059_m32_ft32_smallonly.py"
DESTINATION = ROOT / "submission_s060_m32_ft32_guarded.py"
SOURCE_SHA256 = "b3b3df091059563740b0aff76fef1c235141c5592a80d0ac4ba209620e30b8c6"
CUDA_PREFIX = 'MOE_CUDA_SRC = r"""\n'
CUDA_SUFFIX = '\n"""\nMOE_CPP_SRC = r"""'


source_bytes = SOURCE.read_bytes()
actual_source = hashlib.sha256(source_bytes).hexdigest()
if actual_source != SOURCE_SHA256:
    raise RuntimeError(f"unexpected S059 hash: {actual_source}")
candidate = source_bytes.decode("utf-8")

cuda_start = candidate.index(CUDA_PREFIX) + len(CUDA_PREFIX)
cuda_end = candidate.index(CUDA_SUFFIX, cuda_start)
cuda = candidate[cuda_start:cuda_end]
old = """      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;"""
new = """      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#if OKC_FT_T > 8
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif
#if OKC_FT_T > 32
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif"""
count = cuda.count(old)
if count != 1:
    raise RuntimeError(f"WGS switch: expected one match, found {count}")
cuda = cuda.replace(old, new, 1)
candidate = candidate[:cuda_start] + cuda + candidate[cuda_end:]

for fragment in ("#if OKC_FT_T > 8", "#if OKC_FT_T > 32", 'key, defines = "b4_ft32_small"'):
    if fragment not in candidate:
        raise RuntimeError(f"missing S060 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S060 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
