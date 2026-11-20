"""Generate an exact q8 V128 candidate with streaming L2 eviction priority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
source = ROOT / "submission_sweep_l2kv_k64_v128.py"
text = source.read_text(encoding="utf-8")

old = '''  else asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\\n" ::"r"(dst), "l"(src));'''
new = '''  else {
    // V rows are consumed once.  Keep the measured 128-byte prefetch while marking the line streaming;
    // both qualifiers are cache hints only and copy the same exact 16 bytes.
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;\\n" : "=l"(policy));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\\n"
                 ::"r"(dst), "l"(src), "l"(policy));
  }'''

if text.count(old) != 1:
    raise SystemExit(f"V128 anchor changed: {text.count(old)}")

text = text.replace(old, new, 1).replace(
    'name="okc_attn_v298_q1occ3_l2kv_k64_v128_90a"',
    'name="okc_attn_v298_q1occ3_v128_evictfirst_90a"',
    1,
)
out = ROOT / "submission_sweep_v128_evictfirst.py"
out.write_text(text, encoding="utf-8", newline="\n")
print(out)
