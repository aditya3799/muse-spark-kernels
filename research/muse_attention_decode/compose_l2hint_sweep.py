"""Generate exact q1/q8-specialized cp.async L2 prefetch-hint variants."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission.py"
base = SOURCE.read_text(encoding="utf-8")

old_helper = '''__device__ __forceinline__ void cp_async16_l2_64(uint32_t dst, const void* src) {
  // The flat page-1 path has eight lanes fetch the eight 16-byte chunks of one aligned
  // 128-byte cache row.  Match CUTLASS's cache-global copy primitive and tell L2 the
  // transaction footprint; this is a performance hint and does not change semantics.
  asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}'''
call = "cp_async16_l2_64("

if base.count(old_helper) != 1 or base.count(call) != 3:
    raise SystemExit(f"L2 anchors changed: helper={base.count(old_helper)} calls={base.count(call)}")

variants = {
    "q1_128_q8_64": ("128B", "64B"),
    "q1_64_q8_128": ("64B", "128B"),
    "q1_64_q8_256": ("64B", "256B"),
    "q1_none_q8_64": (None, "64B"),
    "q1_64_q8_none": ("64B", None),
}

for tag, (h1, h8) in variants.items():
    def inst(hint: str | None) -> str:
        suffix = f".L2::{hint}" if hint else ""
        return f'asm volatile("cp.async.cg.shared.global{suffix} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));'

    new_helper = f'''template <int L>
__device__ __forceinline__ void cp_async16_l2_tuned(uint32_t dst, const void* src) {{
  // Shape-specialized performance hint only; all variants copy the same exact 16 bytes.
  if constexpr (L == 1) {{ {inst(h1)} }}
  else {{ {inst(h8)} }}
}}'''
    text = base.replace(old_helper, new_helper, 1).replace(call, "cp_async16_l2_tuned<L>(")
    text = text.replace(
        'name="okc_attn_v298_q1occ3_np128_90a"',
        f'name="okc_attn_v298_q1occ3_l2_{tag}_90a"',
        1,
    )
    out = ROOT / f"submission_sweep_l2_{tag}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)
