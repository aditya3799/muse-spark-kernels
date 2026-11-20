"""Generate v176: exact-shape seeded Lt plans plus the 2%-class QKV16 route.

The broad v173 sweep identified three deterministic algo-66 schedules with
useful cold-L2 signal.  Rebuild only those exact plans, validate them directly,
and dispatch them only at their exact scored shapes.  This avoids the v173
nearby-shape leakage.  Also admit the exact two-CTA QKV16 plan at a 1% gate;
it repeated as the sole positive RS16 schedule in v175.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs_tournament_v175.py"
OUTPUT = ROOT / "submission_seeded_wide_v176.py"


text = BASE.read_text()
if text.count("_V175_ROUTE_RATIO = 0.97") != 1:
    raise RuntimeError("unexpected v175 route gate")
text = text.replace("_V175_ROUTE_RATIO = 0.97", "_V175_ROUTE_RATIO = 0.99", 1)

text += r'''


# --------------------------------------------- v176: exact-shape deterministic cuBLASLt plans.
# (transposed weight, tile, custom option, cluster encoding, core-time gate)
_V176_SPECS = {
    ("o", 128): (False, 18, 4, 2, 0.97),
    ("o", 256): (False, 20, 6, 2, 0.98),
    ("pooler", 256): (True, 17, 1, 2, 0.98),
}
_v176_base_prepare = prepare
_v176_base_kernel_fn = kernel_fn


def _v176_lt_call(state, key, x, out=None):
    plan = state["v176_plans"][key]
    layer, m = key
    if out is None:
        n = state["rq"]["t"][layer].shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    weight = state["rq"]["t"][layer] if plan.transposed else state["raw"].get(layer)
    if weight is None or not plan.run(x, weight, out):
        raise RuntimeError("v176 seeded Lt launch failed for %s m%d" % key)
    return out


def _v176_validate(state, key, weight):
    layer, m = key
    n, k = weight.shape
    cols = (0, k // 3, (2 * k) // 3, k - 1)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v176_lt_call(state, key, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want)
    exact = exact and bool((out[len(cols)] == 0).all())

    gen = torch.Generator(device=weight.device).manual_seed(176000 + m + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v176_lt_call(state, key, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    ok = exact and bool(torch.isfinite(out).all()) and rel <= 0.01 and worst <= 0.05
    print("v176 exact", layer, m, "one_hot", exact, "fp64", rel, worst,
          "valid", ok, flush=True)
    return ok


def prepare(weights):
    state = _v176_base_prepare(weights)
    state["v176_enabled"] = set()
    state["v176_plans"] = {}
    rq = state.get("rq")
    if not rq or "api" not in rq:
        return state

    dev = weights["pooler"].device
    try:
        props = torch.cuda.get_device_properties(dev)
        if props.major != 9 or "H200" not in props.name:
            return state
        api = rq["api"]
        for key, (tr, tile, custom, cluster, gate) in _V176_SPECS.items():
            layer, m = key
            n, k = weights[layer].shape
            try:
                state["v176_plans"][key] = rq_LtPlan(
                    api, m, n, k, tr, tile, custom, False,
                    cluster=cluster, seed=True)
            except Exception as exc:
                print("v176 plan unavailable", layer, m, type(exc).__name__, str(exc), flush=True)

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(176256)
        for key, (_, _, _, _, gate) in _V176_SPECS.items():
            if key not in state["v176_plans"]:
                continue
            layer, m = key
            weight = weights[layer]
            if not _v176_validate(state, key, weight):
                continue
            x = torch.randn((m, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": m, "n": weight.shape[0], "k": weight.shape[1]}
            data = {"x": x, "config": cfg}
            out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda key=key, x=x, out=out: _v176_lt_call(state, key, x, out),
                lambda data=data: _v176_base_kernel_fn(data, state),
                flush, repeats=13, rounds=3,
            )
            keep = candidate_us < incumbent_us and ratio <= gate
            if keep:
                state["v176_enabled"].add(key)
            print("v176 seeded", layer, m, "candidate", candidate_us,
                  "incumbent", incumbent_us, "core_ratio", ratio,
                  "gate", gate, "qualified", keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v176 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["layer"], cfg["m"])
    if key in state.get("v176_enabled", ()):
        return _v176_lt_call(state, key, data["x"])
    return _v176_base_kernel_fn(data, state)
'''

compile(text, str(OUTPUT), "exec")

if __name__ == "__main__":
    OUTPUT.write_text(text)
    print(OUTPUT)
