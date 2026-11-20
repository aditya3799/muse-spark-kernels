"""Generate v279: complete v221 plus per-layer winning pair-plane policies.

O/m1 keeps the ordinary read-only-cache loads that won in v248.  Unpooler/m1
uses the L1-no-allocate weight loads that won in v257.  Unlike the earlier
isolates, the final extension preserves the v221 sk2s/sk2_l2 entry points so
all protected pinned routes remain available.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_weight_stream_v257.py"
OUTPUT = ROOT / "submission_plain_complete_pairplanes_v279.py"

text = BASE.read_text(encoding="utf-8")

helper = r'''
template <int K>
__device__ __forceinline__ uint4 v279_pair_ld4(const uint4* p) {
  if constexpr (K == 4096) return v257_pair_ld4_na(p);
  return __ldg(p);
}
'''
anchor = "__device__ __forceinline__ uint4 v257_pair_ld4_na(const uint4* p) {"
pos = text.find(anchor)
if pos < 0:
    raise RuntimeError("v257 helper marker missing")
end = text.find("\n}\n", pos)
if end < 0:
    raise RuntimeError("v257 helper end missing")
end += 3
text = text[:end] + helper + text[end:]

for old in (
        "v257_pair_ld4_na(pv + q)",
        "v257_pair_ld4_na(pv + CHUNKS + q)",
        "v257_pair_ld4_na(pv + 2 * CHUNKS + q)"):
    if text.count(old) != 1:
        raise RuntimeError(f"weight-load marker mismatch: {old}")
    text = text.replace(old, old.replace("v257_pair_ld4_na", "v279_pair_ld4<K>"), 1)

old = """        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC +
                    _rk_src(sm90))
        _ext = load_inline("""
new = """        sk2s = _v126_sk2s_src(sm90)
        sk2l = _v134_sk2_l2_src(sm90)
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    sk2s + sk2l + _gq_src(sm90) + _gqx_src(sm90) +
                    RGEMV_SRC + _rk_src(sm90))
        _ext = load_inline("""
if text.count(old) < 1:
    raise RuntimeError("final extension source marker mismatch")
pos = text.rfind(old)
text = text[:pos] + new + text[pos + len(old):]

old = """            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq",
                       "gqx", "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "cgemv_pair", "cgemv_pair_info", "rk"],"""
new = """            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "sk2s",
                       "sk2s_info", "sk2_l2", "sk2_l2_info", "gq", "gqx",
                       "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "cgemv_pair", "cgemv_pair_info", "rk"],"""
if text.count(old) < 1:
    raise RuntimeError("final extension function marker mismatch")
pos = text.rfind(old)
text = text[:pos] + new + text[pos + len(old):]

text = text.replace("okc_bf16proj_v257_c12pair_",
                    "okc_bf16proj_v279_completepair_", 1)
text = text.replace("v257 c12 pairplanes", "v279 complete pairplanes")

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
