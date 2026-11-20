"""Build S030: validated S029 v81 verify attention plus S028's M32 MoE tile.

Both parents branch from the promoted S022 source and touch disjoint regions.
This builder applies only the six audited M32 substitutions to S029, preserving
the validated v81 integration byte-for-byte.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s029_v81_qkv_shape_fix.py"
DESTINATION = ROOT / "submission_s030_v81_m32.py"
SOURCE_SHA256 = "06ab5a52fff0a61b1f232553ff2ea2473177918517bd5c4811e4244819c6cacd"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_bytes = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(
        f"unexpected S029 source hash: {actual_sha256} != {SOURCE_SHA256}"
    )
candidate = source_bytes.decode("utf-8").replace("\r\n", "\n")

candidate = replace_once(
    candidate,
    "#define SLOTB (PPS * PC)\n#define ZB (2048 + 64)",
    "#define SLOTB (PPS * PC)\n"
    "#define REDB (SW == 1 ? 0 : NW * 2 * 32 * 16)  // SW=1 consumes fragments directly; no reduction slab\n"
    "#define ZB (2048 + 64)",
    "conditional reduction slab size",
)

old_reduction = """#pragma unroll
  for (int nb = 0; nb < NT; nb += 2) {
#pragma unroll
    for (int n = nb; n < nb + 2 && n < NT; n++)
      red[(wid * 2 + n - nb) * 32 + lane] = make_float4(acc[n][0][0] + acc[n][1][0], acc[n][0][1] + acc[n][1][1], acc[n][0][2] + acc[n][1][2], acc[n][0][3] + acc[n][1][3]);
    gbar(1 + grp, SW * 32);
    if (lane < 2 * (32 / SW)) {
      const int n = lane / (32 / SW), f = s * (32 / SW) + lane % (32 / SW);
      if (nb + n < NT) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < SW; w++) { const float4 u = red[((grp * SW + w) * 2 + n) * 32 + f]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
        out(nb + n, f, v);
      }
    }
    gbar(1 + grp, SW * 32);  // partials consumed before they are overwritten
  }
"""
new_reduction = """  if constexpr (SW == 1) {
    // One warp owns the complete K slice: consume its fragments directly.
    // This also avoids named barrier 1 + grp reaching the illegal ID 16.
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const float4 v = make_float4(acc[n][0][0] + acc[n][1][0], acc[n][0][1] + acc[n][1][1],
                                   acc[n][0][2] + acc[n][1][2], acc[n][0][3] + acc[n][1][3]);
      out(n, lane, v);
    }
  } else {
#pragma unroll
    for (int nb = 0; nb < NT; nb += 2) {
#pragma unroll
      for (int n = nb; n < nb + 2 && n < NT; n++)
        red[(wid * 2 + n - nb) * 32 + lane] = make_float4(acc[n][0][0] + acc[n][1][0], acc[n][0][1] + acc[n][1][1], acc[n][0][2] + acc[n][1][2], acc[n][0][3] + acc[n][1][3]);
      gbar(1 + grp, SW * 32);
      if (lane < 2 * (32 / SW)) {
        const int n = lane / (32 / SW), f = s * (32 / SW) + lane % (32 / SW);
        if (nb + n < NT) {
          float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int w = 0; w < SW; w++) { const float4 u = red[((grp * SW + w) * 2 + n) * 32 + f]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
          out(nb + n, f, v);
        }
      }
      gbar(1 + grp, SW * 32);  // partials consumed before they are overwritten
    }
  }
"""
candidate = replace_once(
    candidate, old_reduction, new_reduction, "one-warp fragment consumption"
)
candidate = replace_once(
    candidate,
    "const int zo = rdo + NW * 2 * 32 * 16;",
    "const int zo = rdo + REDB;",
    "device shared-memory layout",
)
candidate = replace_once(
    candidate,
    "xo = ro + NW * SG * SLOTB + NW * 2 * 32 * 16 + ZB;",
    "xo = ro + NW * SG * SLOTB + REDB + ZB;",
    "host shared-memory layout",
)
candidate = replace_once(
    candidate,
    'lambda: M__extK("-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),',
    'lambda: M__extK("-DSW=1 -DKS1=2 -DKS2=4 -DMAXM=32"),',
    "M32 prepare specialization",
)
candidate = replace_once(
    candidate,
    '32: ("gk", 0, 0, "-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),',
    '32: ("gk", 0, 0, "-DSW=1 -DKS1=2 -DKS2=4 -DMAXM=32"),',
    "M32 dispatch specialization",
)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S030 exceeds the challenge source limit")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
