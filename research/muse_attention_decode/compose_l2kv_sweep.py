"""Generate exact q8 K/V-specific cp.async L2 prefetch-hint variants."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
base = (ROOT / "submission.py").read_text(encoding="utf-8")

old_helper = '''__device__ __forceinline__ void cp_async16_l2_64(uint32_t dst, const void* src) {
  // The flat page-1 path has eight lanes fetch the eight 16-byte chunks of one aligned
  // 128-byte cache row.  Match CUTLASS's cache-global copy primitive and tell L2 the
  // transaction footprint; this is a performance hint and does not change semantics.
  asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}'''
k_call = "cp_async16_l2_64(kb + pso + k * (PR * ROWB), p.k_cache + off);"
v_call = "cp_async16_l2_64(vb + pso + k * (PR * ROWB), p.v_cache + off);"

if base.count(old_helper) != 1 or base.count(k_call) != 1 or base.count(v_call) != 1:
    raise SystemExit("K/V L2 anchors changed")

variants = {
    "k128_v64": ("64B", "64B", "128B", "64B"),
    "k64_v128": ("64B", "64B", "64B", "128B"),
    "k128_v256": ("64B", "64B", "128B", "256B"),
    "k256_v128": ("64B", "64B", "256B", "128B"),
    "all_k64_v128": ("64B", "128B", "64B", "128B"),
    "q1_k128_v64_q8_k64_v128": ("128B", "64B", "64B", "128B"),
}

for tag, (kh1, vh1, kh8, vh8) in variants.items():
    helper = f'''template <int L>
__device__ __forceinline__ void cp_async16_l2_k(uint32_t dst, const void* src) {{
  if constexpr (L == 1) asm volatile("cp.async.cg.shared.global.L2::{kh1} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
  else asm volatile("cp.async.cg.shared.global.L2::{kh8} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}}
template <int L>
__device__ __forceinline__ void cp_async16_l2_v(uint32_t dst, const void* src) {{
  if constexpr (L == 1) asm volatile("cp.async.cg.shared.global.L2::{vh1} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
  else asm volatile("cp.async.cg.shared.global.L2::{vh8} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}}'''
    text = base.replace(old_helper, helper, 1)
    text = text.replace(k_call, k_call.replace("cp_async16_l2_64", "cp_async16_l2_k<L>"), 1)
    text = text.replace(v_call, v_call.replace("cp_async16_l2_64", "cp_async16_l2_v<L>"), 1)
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_l2kv_{tag}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_l2kv_{tag}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
