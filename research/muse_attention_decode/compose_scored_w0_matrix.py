"""Generate the fourteen exact scored refinements of the active q8-W0 winner."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = (ROOT / "submission.py").read_text(encoding="utf-8")
MODULE = 'name="okc_attn_v298_q1occ3_np128_q8w0_90a"'
W0 = "OKC_W0_L8 = 1135"
W0_EXPR = "(OKC_W0_L1 if L == 1 else OKC_W0_L8)"
SEG = "#define OKC_SEGK8 600"


def replace_one(text, old, new, tag):
    if text.count(old) != 1:
        raise SystemExit(f"{tag}: anchor {old!r} count {text.count(old)}")
    return text.replace(old, new, 1)


def emit(tag, text):
    text = replace_one(text, MODULE, f'name="okc_attn_v298_score_{tag}_90a"', tag)
    out = ROOT / f"submission_score_{tag}.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(out)


for value in (1120, 1125, 1130, 1138, 1140, 1145, 1150):
    tag = f"w0_{value}"
    emit(tag, replace_one(BASE, W0, f"OKC_W0_L8 = {value}", tag))

for b16, b32 in ((1130, 1140), (1140, 1130), (1135, 1140), (1140, 1135)):
    tag = f"w0_b16_{b16}_b32_{b32}"
    text = replace_one(
        BASE,
        W0,
        f"OKC_W0_L8 = 1135\nOKC_W0_L8_B16 = {b16}\nOKC_W0_L8_B32 = {b32}",
        tag,
    )
    text = replace_one(
        text,
        W0_EXPR,
        "(OKC_W0_L1 if L == 1 else (OKC_W0_L8_B16 if B <= 16 else OKC_W0_L8_B32))",
        tag,
    )
    emit(tag, text)

for value in (550, 650):
    tag = f"w0_segk8_{value}"
    emit(tag, replace_one(BASE, SEG, f"#define OKC_SEGK8 {value}", tag))

old_helper = '''__device__ __forceinline__ void cp_async16_l2_64(uint32_t dst, const void* src) {
  // The flat page-1 path has eight lanes fetch the eight 16-byte chunks of one aligned
  // 128-byte cache row.  Match CUTLASS's cache-global copy primitive and tell L2 the
  // transaction footprint; this is a performance hint and does not change semantics.
  asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}'''
k_call = "cp_async16_l2_64(kb + pso + k * (PR * ROWB), p.k_cache + off);"
v_call = "cp_async16_l2_64(vb + pso + k * (PR * ROWB), p.v_cache + off);"

for tag, kh8, vh8 in (("w0_l2_k64_v128", "64B", "128B"), ("w0_l2_k128_v128", "128B", "128B")):
    helper = f'''template <int L>
__device__ __forceinline__ void cp_async16_l2_k(uint32_t dst, const void* src) {{
  if constexpr (L == 1) asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
  else asm volatile("cp.async.cg.shared.global.L2::{kh8} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}}
template <int L>
__device__ __forceinline__ void cp_async16_l2_v(uint32_t dst, const void* src) {{
  if constexpr (L == 1) asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
  else asm volatile("cp.async.cg.shared.global.L2::{vh8} [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));
}}'''
    text = replace_one(BASE, old_helper, helper, tag)
    text = replace_one(text, k_call, k_call.replace("cp_async16_l2_64", "cp_async16_l2_k<L>"), tag)
    text = replace_one(text, v_call, v_call.replace("cp_async16_l2_64", "cp_async16_l2_v<L>"), tag)
    emit(tag, text)
