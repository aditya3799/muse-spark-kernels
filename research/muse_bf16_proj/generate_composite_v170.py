from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs_small_v158.py"
REFERENCE = ROOT / "submission_qkv256_lt124_v92.py"
OUTPUT = ROOT / "submission_composite_v170.py"


text = BASE.read_text()
ref = REFERENCE.read_text()

# Construct the exact algo-66 / tile-124 / cluster-2 QKV-256 plan.  The plain
# tile-number edit in v168 did not reproduce this schedule because the cluster
# attribute and clean algo initialization are both significant.
old_sig = '            "cublasLtDestroy": [V],\n'
new_sig = old_sig + '            "cublasLtMatmulAlgoInit": [V, I, I, I, I, I, I, I, _ct.POINTER(rq_LtAlgo)],\n'
if text.count(old_sig) != 1:
    raise RuntimeError("unexpected v158 cuBLASLt signature table")
text = text.replace(old_sig, new_sig, 1)

start = "class rq_LtPlan:\n"
end = "\n\nRQ_LT = ["
base_a, base_b = text.index(start), text.index(end, text.index(start))
ref_a, ref_b = ref.index(start), ref.index(end, ref.index(start))
text = text[:base_a] + ref[ref_a:ref_b] + text[base_b:]

old_route = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 134, 2, True), ("o", 256, False, 20, 4, False),'
new_route = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 124, 0, False), ("o", 256, False, 20, 4, False),'
if text.count(old_route) != 1:
    raise RuntimeError("unexpected v158 qkv_gate m256 route")
text = text.replace(old_route, new_route, 1)

old_plan = '                rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)\n'
new_plan = (
    '                if layer == "qkv_gate" and m == 256:\n'
    '                    rq["plans"][(layer, m)] = rq_LtPlan(\n'
    '                        api, m, n, k, False, 124, 0, False, cluster=2, seed=True)\n'
    '                else:\n'
    '                    rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)\n'
)
if text.count(old_plan) != 1:
    raise RuntimeError("unexpected v158 rq plan construction")
text = text.replace(old_plan, new_plan, 1)

# The isolated v146 H200 check showed that issuing all eight router K-split
# slices up front is faster at m=32.  Patch only that configuration inside the
# alternate module; leave the rest of the current v150/v158 lineage intact.
old_exec = 'exec(compile(_zl.decompress(_b64.b64decode(_ALT_SRC)).decode(), "_alt_proj", "exec"), _alt.__dict__)\n'
new_exec = (
    '_ALT_DECODED = _zl.decompress(_b64.b64decode(_ALT_SRC)).decode()\n'
    '_R32_BASE = "ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]"\n'
    '_R32_ST8 = "ROUTER_CONFIGS = [(32,16,8,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]"\n'
    'if _ALT_DECODED.count(_R32_BASE) != 1:\n'
    '    raise RuntimeError("Router m32 baseline config was not found exactly once")\n'
    '_ALT_DECODED = _ALT_DECODED.replace(_R32_BASE, _R32_ST8)\n'
    'exec(compile(_ALT_DECODED, "_alt_proj", "exec"), _alt.__dict__)\n'
)
if text.count(old_exec) != 1:
    raise RuntimeError("unexpected v158 alternate-module loader")
text = text.replace(old_exec, new_exec, 1)

# Only QKV-32 was a robust winner in v158 (5.85% core-time reduction).  Avoid
# compiling and validating the seven rejected small-shape experiments, then
# allow the already strict per-route 3% paired gate to enable that winner.
old_routes = '''_V158_ROUTES = {
    (32, "qkv_gate"): 1,
    (32, "pooler"): 1,
    (32, "unpooler"): 1,
    (32, "o"): 2,
    (16, "qkv_gate"): 3,
    (16, "pooler"): 4,
    (16, "unpooler"): 4,
    (16, "o"): 4,
}'''
new_routes = '''_V158_ROUTES = {
    (32, "qkv_gate"): 1,
}'''
if text.count(old_routes) != 1:
    raise RuntimeError("unexpected v158 route table")
text = text.replace(old_routes, new_routes, 1)

if text.count('_V158_SCORE_GATE = 0.009') != 1:
    raise RuntimeError("unexpected v158 aggregate gate")
text = text.replace('_V158_SCORE_GATE = 0.009', '_V158_SCORE_GATE = 0.0', 1)

OUTPUT.write_text(text)
print(OUTPUT)
