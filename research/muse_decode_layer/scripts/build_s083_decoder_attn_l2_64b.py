"""Build S083: port the standalone attention page-1 L2::64B copy hint.

This is intentionally a one-mechanism decoder experiment. It adds the exact
helper used by the scored standalone attention winner and applies it only to
the two K/V copies in the flat page-size-1 fast path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
OUT = ROOT / "submission_s083_attn_l2_64b.py"

BASE_SHA256 = "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("active decoder hash mismatch; refusing to extend")

base = BASE.read_text(encoding="utf-8")

old_helper = '''__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
'''
new_helper = old_helper + '''__device__ __forceinline__ void cp_async16_l2_64(uint32_t dst, const void* src) {
  // Flat page-1: eight lanes fetch the eight aligned 16-byte chunks of a
  // 128-byte cache row. Expose that transaction footprint to L2.
  asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}
'''
if base.count(old_helper) != 1:
    raise RuntimeError(f"attention helper: expected 1 match, found {base.count(old_helper)}")
base = base.replace(old_helper, new_helper, 1)

old_calls = '''            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
'''
new_calls = '''            cp_async16_l2_64(kb + pso + k * (PR * ROWB), p.k_cache + off);
            cp_async16_l2_64(vb + pso + k * (PR * ROWB), p.v_cache + off);
'''
if base.count(old_calls) != 1:
    raise RuntimeError(f"page-1 copy site: expected 1 match, found {base.count(old_calls)}")
candidate = base.replace(old_calls, new_calls, 1)

compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode("utf-8"))
print(OUT)
print(hashlib.sha256(candidate.encode("utf-8")).hexdigest())
