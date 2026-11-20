"""Generate v173: deterministic cuBLASLt schedule sweep on the wide BF16 routes.

Keep the validated v150 RS128 paths and v158 QKV32 path, reconstruct the
validated tile-124/cluster-2 QKV256 schedule, then expose a bounded set of
clean algo-66 schedules for the remaining O/Pooler/Unpooler wide calls.  The
existing cold-L2 CUDA-graph autotuner may promote a new schedule only when its
fixed-overhead-subtracted median is at least 3% faster than every non-sweep
candidate for that exact shape.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs_small_v158.py"
REFERENCE = ROOT / "submission_qkv256_lt124_v92.py"
OUTPUT = ROOT / "submission_lt_seed_sweep_v173.py"


text = BASE.read_text()
ref = REFERENCE.read_text()

# Use AlgoInit so every tested schedule starts from the same clean algo-66
# state.  This also reconstructs the validated QKV256 tile-124/cluster-2 plan.
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
    raise RuntimeError("unexpected v158 QKV256 route")
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
    raise RuntimeError("unexpected v158 plan construction")
text = text.replace(old_plan, new_plan, 1)

# Two neighboring tile shapes, all eight custom options, and the default plus
# four compact cluster encodings.  Invalid configurations are discarded by
# cublasLtMatmulAlgoCheck.  These are only four scored routes, so the bounded
# sweep remains comfortably inside the prepare budget.
old_tma = 'RQ_TMA = {("pooler", 128): (64, 64, 128, 6),'
seed_block = '''RQ_LT_V173 = [
    (layer, m, tr, tile, custom, cluster)
    for layer, m, tr, tiles in (
        ("o", 128, False, (17, 18)),
        ("o", 256, False, (19, 20)),
        ("pooler", 256, True, (17, 18)),
        ("unpooler", 256, True, (19, 20)),
    )
    for tile in tiles for custom in range(8) for cluster in range(5)
]


def _v173_lt_name(layer, m, tile, custom, cluster):
    return "rq_lt_seed_%s_%d_t%dc%dk%d" % (
        layer, m, tile, custom, cluster)


RQ_TMA = {("pooler", 128): (64, 64, 128, 6),'''
if text.count(old_tma) != 1:
    raise RuntimeError("unexpected RQ_TMA anchor")
text = text.replace(old_tma, seed_block, 1)

old_state = '    rq = {"t": {}, "plans": {}}\n'
new_state = '    rq = {"t": {}, "plans": {}, "extra_plans": {}}\n'
if text.count(old_state) != 1:
    raise RuntimeError("unexpected rq state")
text = text.replace(old_state, new_state, 1)

plan_tail = new_plan + '            except Exception:\n                pass\n'
extra_build = plan_tail + '''        for layer, m, tr, tile, custom, cluster in RQ_LT_V173:
            n, k = weights[layer].shape
            name = _v173_lt_name(layer, m, tile, custom, cluster)
            try:
                rq["extra_plans"][name] = rq_LtPlan(
                    api, m, n, k, tr, tile, custom, False,
                    cluster=cluster, seed=True)
            except Exception:
                pass
'''
if text.count(plan_tail) != 1:
    raise RuntimeError("unexpected rq prepare loop")
text = text.replace(plan_tail, extra_build, 1)

old_candidates = '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c
'''
new_candidates = '''    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    prefix = "rq_lt_seed_%s_%d_" % (layer, m)
    c.extend((name,) for name in sorted(rq.get("extra_plans", {}))
             if name.startswith(prefix))
    return c
'''
if text.count(old_candidates) != 1:
    raise RuntimeError("unexpected rq candidate block")
text = text.replace(old_candidates, new_candidates, 1)

old_dispatch = '    if kind in ("rq_lt", "rq_tma"):\n'
new_dispatch = '    if kind == "rq_tma" or kind.startswith("rq_lt"):\n'
if text.count(old_dispatch) != 1:
    raise RuntimeError("unexpected _run rq dispatch")
text = text.replace(old_dispatch, new_dispatch, 1)

old_lt_run = '''    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
'''
new_lt_run = '''    if choice[0].startswith("rq_lt"):
        plan = (rq["plans"].get((layer, m)) if choice[0] == "rq_lt"
                else rq.get("extra_plans", {}).get(choice[0]))
'''
if text.count(old_lt_run) != 1:
    raise RuntimeError("unexpected rq_run block")
text = text.replace(old_lt_run, new_lt_run, 1)

# A sweep candidate must beat the best incumbent by >=3% after subtracting the
# same approximate 4 us graph replay overhead used by the official harness.
old_best = '''            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
'''
new_best = '''            best = min(range(len(cands)), key=lambda i: med[i])
            if cands[best][0].startswith("rq_lt_seed_"):
                incumbent = [i for i, ch in enumerate(cands)
                             if not ch[0].startswith("rq_lt_seed_")]
                if incumbent:
                    ib = min(incumbent, key=lambda i: med[i])
                    cand_core = max(1000.0 * med[best] - 4.0, 1e-6)
                    base_core = max(1000.0 * med[ib] - 4.0, 1e-6)
                    ratio = cand_core / base_core
                    print("v173 LT gate", layer, m, cands[best][0],
                          "candidate", 1000.0 * med[best],
                          "incumbent", 1000.0 * med[ib],
                          "core_ratio", ratio, flush=True)
                    if ratio > 0.97:
                        best = ib
            state["choice"][(layer, m)] = cands[best]
'''
if text.count(old_best) != 1:
    raise RuntimeError("unexpected autotune best block")
text = text.replace(old_best, new_best, 1)

# Compile only the single robust v158 QKV32 compressed route.
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
