"""Build S057: S056 joined completion plus dual WGMMA accumulators for NTT=1."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s056_joined_wgs_completion.py"
DESTINATION = ROOT / "submission_s057_wgs_dual_accumulator.py"
SOURCE_SHA256 = "e19c8b10fd6585b9c2d69e6c40efc3584036ea1ed57f3a0b982806c260b1751d"
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
    raise RuntimeError(f"unexpected S056 hash: {actual_source}")
candidate = source_bytes.decode("utf-8")

cuda_start = candidate.index(CUDA_PREFIX) + len(CUDA_PREFIX)
cuda_end = candidate.index(CUDA_SUFFIX, cuda_start)
cuda = candidate[cuda_start:cuda_end]

cuda = replace_once(
    cuda,
    """  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)""",
    """  float d[WGS_NT][4];
  float d_odd[WGS_NT][4];  // NTT=1: odd commit groups use an independent dependency chain""",
    "dual accumulator declaration",
)
cuda = replace_once(
    cuda,
    """        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }""",
    """        for (int jj = 0; jj < GS; ++jj) {
          if constexpr(NTT == 1) {
            if (ci & 1) {
              if (FIRST && ci == 1 && jj == 0)
                WGSMMA<NTT>::mma0(d_odd, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                                  xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
              else
                WGSMMA<NTT>::mma(d_odd, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                                 xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
            } else {
              if (FIRST && ci == 0 && jj == 0)
                WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
              else
                WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                                 xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
            }
          } else {
            if (FIRST && ci == 0 && jj == 0)
              WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
            else
              WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                               xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
          }
        }""",
    "alternating WGMMA issue",
)
cuda = replace_once(
    cuda,
    """  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);""",
    """  wg_wait<0>();
  if constexpr(NTT == 1) {
#pragma unroll
    for (int q = 0; q < 4; ++q) d[0][q] += d_odd[0][q];
  }
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);""",
    "one-tile accumulator combine",
)

candidate = candidate[:cuda_start] + cuda + candidate[cuda_end:]
required = (
    "float d_odd[WGS_NT][4]",
    "if constexpr(NTT == 1)",
    "d[0][q] += d_odd[0][q]",
    "Joined completion requires equal ring depths",
)
for fragment in required:
    if fragment not in candidate:
        raise RuntimeError(f"missing S057 fragment: {fragment}")

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError(f"S057 exceeds source limit: {len(encoded):,}")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
