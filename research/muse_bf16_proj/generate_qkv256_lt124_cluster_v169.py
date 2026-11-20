from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
REFERENCE = ROOT / "submission_qkv256_lt124_v92.py"
OUTPUT = ROOT / "submission_qkv256_lt124_cluster_v169.py"


text = BASE.read_text()
ref = REFERENCE.read_text()

# v92 contains the already validated ctypes definitions for constructing an
# algo-66 plan from scratch and setting Hopper's uint16 cluster-shape field.
old_sig = '            "cublasLtDestroy": [V],\n'
new_sig = old_sig + '            "cublasLtMatmulAlgoInit": [V, I, I, I, I, I, I, I, _ct.POINTER(rq_LtAlgo)],\n'
if text.count(old_sig) != 1:
    raise RuntimeError("unexpected v150 cuBLASLt signature table")
text = text.replace(old_sig, new_sig, 1)

start = "class rq_LtPlan:\n"
end = "\n\nRQ_LT = ["
base_a, base_b = text.index(start), text.index(end, text.index(start))
ref_a, ref_b = ref.index(start), ref.index(end, ref.index(start))
text = text[:base_a] + ref[ref_a:ref_b] + text[base_b:]

old_route = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 134, 2, True), ("o", 256, False, 20, 4, False),'
new_route = '("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 124, 0, False), ("o", 256, False, 20, 4, False),'
if text.count(old_route) != 1:
    raise RuntimeError("unexpected v150 qkv_gate m256 route")
text = text.replace(old_route, new_route, 1)

old_plan = '                rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)\n'
new_plan = (
    '                if layer == "qkv_gate" and m == 256:\n'
    '                    # Exact nvjet 72x64 plan: algo 66, tile 124, stage 35, cluster shape 2.\n'
    '                    rq["plans"][(layer, m)] = rq_LtPlan(\n'
    '                        api, m, n, k, False, 124, 0, False, cluster=2, seed=True)\n'
    '                else:\n'
    '                    rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)\n'
)
if text.count(old_plan) != 1:
    raise RuntimeError("unexpected v150 rq plan construction")
text = text.replace(old_plan, new_plan, 1)

OUTPUT.write_text(text)
print(OUTPUT)
