"""Build N72 + explicit BF16 cuBLASLt plans on the pre-route-shaping K64 source."""

import ast
from pathlib import Path


root = Path(__file__).resolve().parent.parent
base = (root / "submission_fastaccum_d52_control.py").read_text()
donor = (root / "submission_keell0renz_fair_base.py").read_text()
out = root / "submission_n72_lt2_exactroute_bf16_candidate.py"


def between(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


# The ancestor predates route shaping and O-FP8.  Force its optional pooler/unpooler FP8 experiment off.
old = "UNPOOL_FP8 = True  # experimental: lossy E4M3 unpooler weights/activations (explicit user-authorized screening risk)"
new = "UNPOOL_FP8 = False  # exact BF16 pooler/unpooler only"
assert base.count(old) == 1
base = base.replace(old, new, 1)

# N72 FMHA and dispatch.
start = donor.index("# ---------------------------------------------------------------------------------------------- ZhimaoL fmha7")
end = donor.index("def kernel_fn(data, state):", start)
block = donor[start:end].rstrip() + "\n\n\n"
marker = "def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None):"
assert base.count(marker) == 1
base = base.replace(marker, block + marker, 1)
assert base.count("    if SELFTEST:\n") == 1
base = base.replace("    if SELFTEST:\n", "    _zh_prepare(st)\n    if SELFTEST:\n", 1)
kpos = base.index("def kernel_fn(data, state):")
head, tail = base[:kpos], base[kpos:]
old = "    av = _attn_variant(st, cfg)\n    pre = None\n"
new = "    av = _attn_variant(st, cfg)\n    zh = _zh_ok(st, cfg)\n    pre = None\n"
assert tail.count(old) == 1
tail = tail.replace(old, new, 1)
old = "    split = av is not None and T <= PREP_SPLIT_MAX_T"
new = "    split = not zh and av is not None and T <= PREP_SPLIT_MAX_T"
assert tail.count(old) == 1
tail = tail.replace(old, new, 1)
old = "    if av is not None:\n"
new = "    if zh:\n        og = _zh_attn(st, data, q8, qkv, cfg)\n    elif av is not None:\n"
assert tail.count(old) >= 1
tail = tail.replace(old, new, 1)
base = head + tail

# Explicit-config BF16 cuBLASLt candidates from the public fair source.
pins = between(donor, "LT_PIN2 = {", "LT_ALGO_TOL")
marker = "FINE_MIN_T = 2048"
assert base.count(marker) == 1 and "LT_PIN2 = {" not in base
base = base.replace(marker, pins + marker, 1)
helpers = between(donor, "def _lt_rel_err", "def _tune_lt")
marker = "def _tune_lt(st):"
assert base.count(marker) == 1 and "def _try_lt2" not in base
base = base.replace(marker, helpers + marker, 1)
old = "        pid = lt.lt_plan(T, N, K, f32, st[\"lt_ws\"].numel(), 32)\n"
new = (
    "        c2 = _try_lt2(st, T, name, x, w, out, ref, ref_fn, t_ref, f32)\n"
    "        if c2 is not None:\n"
    "            choice[(T, name)] = c2\n"
    "            del x, out\n"
    "            continue\n" + old
)
assert base.count(old) == 1
base = base.replace(old, new, 1)
old = "    st[\"lt\"].lt_run(ch[0], ch[1], x, w, out, st[\"lt_ws_side\"] if side else st[\"lt_ws\"])\n    return out\n"
new = (
    "    if ch[0] == \"lt2\":\n"
    "        st[\"lt\"].lt2_run(ch[1], x, w, st.get(\"lt2_wt\", {}).get(wn, w), out, st[\"lt_ws_side\"] if side else st[\"lt_ws\"])\n"
    "    else:\n"
    "        st[\"lt\"].lt_run(ch[0], ch[1], x, w, out, st[\"lt_ws_side\"] if side else st[\"lt_ws\"])\n"
    "    return out\n"
)
assert base.count(old) == 1
base = base.replace(old, new, 1)
decl = (
    "int64_t lt2_add(int64_t m, int64_t n, int64_t k, bool out_f32, int64_t lay, std::vector<int64_t> cfg, int64_t ws_bytes);\n"
    "void lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, torch::Tensor ws);\n"
)
marker = "void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws);"
assert base.count(marker) == 1
base = base.replace(marker, decl + marker, 1)
old = 'functions=["lt_plan", "lt_plan_sm", "lt_nalgo", "lt_info", "lt_run"],'
new = 'functions=["lt_plan", "lt_plan_sm", "lt_nalgo", "lt_info", "lt2_add", "lt2_run", "lt_run"],'
assert base.count(old) == 1
base = base.replace(old, new, 1)
cpp = between(donor, "// ---- explicit-config cuBLASLt algorithms", "int64_t lt_plan_sm")
marker = "static cublasLtHandle_t g_lt = nullptr;\n"
assert base.count(marker) == 1 and "struct Lt2Algo" not in base
base = base.replace(marker, marker + cpp, 1)

assert "ROUTE_SHAPING" not in base
assert "O_FP8" not in base
assert "UNPOOL_FP8 = False" in base
for forbidden in ("base64.b64decode", "zlib.decompress", "exec(compile("):
    assert forbidden not in base, forbidden
ast.parse(base)
compile(base, str(out), "exec")
out.write_bytes(base.encode())
print(out.name, len(base.encode()))
