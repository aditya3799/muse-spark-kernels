"""Generate v189: guarded RS128 four-stage versus five-stage tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
OUTPUT = ROOT / "submission_plain_rs128_stage4_v189.py"

text = BASE.read_text(encoding="utf-8")

old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<128, 4, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v187 rq switch anchor count")
text = text.replace(old_switch, new_switch, 1)

block = r'''

# ---------------------------------------------------------------- v189: RS128 depth-4 direct tournament.
# Depth four lowers dynamic shared memory from the protected depth-five
# instance.  It is compared directly with v187's final route, not with an
# older pre-RS incumbent, and can enter dispatch independently per layer.
_V189_TARGETS = ("qkv_gate", "pooler", "unpooler")
_V189_VARIANT = 2
_V189_ROUTE_RATIO = 0.99

_v189_base_prepare = prepare
_v189_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v189_base_prepare(weights)
    state["v189_enabled"] = set()
    state["v189_plans"] = {}
    vstate = state.get("v150")
    if vstate is None:
        return state

    dev = weights["pooler"].device
    try:
        plans = {}
        for layer in _V189_TARGETS:
            source_plan = vstate.get("plans", {}).get(layer)
            if source_plan is None:
                continue
            plan = dict(source_plan)
            plan["variant"] = _V189_VARIANT
            if _v158_validate(vstate, plan, weights[layer], layer, 128):
                plans[layer] = plan
        state["v189_plans"] = plans
        if not plans:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(189128)
        qualified = set()
        total_inc = total_cand = 0.0
        for layer in _V189_TARGETS:
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen,
                            device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 128, "n": weight.shape[0],
                   "k": weight.shape[1]}
            plan = plans.get(layer)
            if plan is None:
                unchanged = _v150_time(
                    lambda x=x, cfg=cfg: _v189_base_kernel_fn(
                        {"x": x, "config": cfg}, state), flush)
                core = max(unchanged - _V150_REPLAY_US, 0.05)
                total_inc += core
                total_cand += core
                continue
            out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16,
                              device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda x=x, out=out, plan=plan: _v158_call(
                    vstate, plan, x, out),
                lambda x=x, cfg=cfg: _v189_base_kernel_fn(
                    {"x": x, "config": cfg}, state),
                flush, repeats=13, rounds=5)
            keep = candidate_us < incumbent_us and ratio <= _V189_ROUTE_RATIO
            if keep:
                qualified.add((128, layer))
            inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
            cand_core = (max(candidate_us - _V150_REPLAY_US, 0.05)
                         if keep else inc_core)
            total_inc += inc_core
            total_cand += cand_core
            print("v189 rs128-stage4", layer, "candidate", candidate_us,
                  "incumbent", incumbent_us, "core_ratio", ratio,
                  "qualified", keep, flush=True)

        state["v189_enabled"] = qualified
        projected = ((total_inc / total_cand) ** (1.0 / 6.0) - 1.0
                     if total_cand else 0.0)
        print("v189 projected_score_gain", projected,
              "enabled", sorted(qualified), flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state["v189_enabled"] = set()
        print("v189 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v189_enabled", ()):
        plan = state.get("v189_plans", {}).get(cfg["layer"])
        if plan is not None:
            return _v158_call(state["v150"], plan, data["x"])
    return _v189_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
