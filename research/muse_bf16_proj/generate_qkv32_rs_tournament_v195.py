"""Generate v195: bounded plain-source QKV/m32 RS schedule tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_qkv1_qkv32_v187.py"
OUTPUT = ROOT / "submission_plain_qkv32_rs_tournament_v195.py"

text = BASE.read_text(encoding="utf-8")

old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 3: return rq_go<32, 6, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
    case 4: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
    case 5: return rq_go<32, 6, 1, 0, 0, 0, 35, 1>(p, x, out, (int)KS, (int)info);
    case 6: return rq_go<32, 5, 1, 0, 0, 0, 35, 1>(p, x, out, (int)KS, (int)info);
    case 7: return rq_go<32, 4, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v187 rq switch anchor count")
text = text.replace(old_switch, new_switch, 1)

block = r'''

# ---------------------------------------------------------------- v195: QKV/m32 exact RS schedules.
_V195_VARIANTS = (2, 3, 4, 5, 6, 7)
_V195_ROUTE_RATIO = 0.99
_v195_base_prepare = prepare
_v195_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v195_base_prepare(weights)
    state["v195_enabled"] = False
    state["v195_plan"] = None
    vstate = state.get("v150")
    source = None if vstate is None else vstate.get("plans", {}).get("qkv_gate")
    if source is None:
        return state

    dev = weights["qkv_gate"].device
    try:
        weight = weights["qkv_gate"]
        m = 32
        gen = torch.Generator(device=dev).manual_seed(195032)
        x = torch.randn((m, weight.shape[1]), generator=gen,
                        device=dev).to(torch.bfloat16)
        cfg = {"layer": "qkv_gate", "m": m, "n": weight.shape[0],
               "k": weight.shape[1]}
        incumbent = lambda: _v195_base_kernel_fn({"x": x, "config": cfg}, state)
        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)

        screened = []
        plans = {}
        outputs = {}
        for variant in _V195_VARIANTS:
            try:
                plan = dict(source)
                plan["variant"] = variant
                if not _v158_validate(vstate, plan, weight, "qkv_gate", m):
                    continue
                out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16,
                                  device=dev)
                fn = lambda plan=plan, out=out: _v158_call(
                    vstate, plan, x, out)
                samples = [_v150_time(fn, flush, repeats=9) for _ in range(3)]
                samples.sort()
                screened.append((samples[1], variant))
                plans[variant] = plan
                outputs[variant] = out
            except Exception as exc:
                print("v195 variant rejected", variant, type(exc).__name__,
                      str(exc)[:240], flush=True)
        screened.sort()
        print("v195 qkv32 screen", screened, flush=True)

        paired = []
        for _, variant in screened[:3]:
            plan, out = plans[variant], outputs[variant]
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda plan=plan, out=out: _v158_call(vstate, plan, x, out),
                incumbent, flush, repeats=13, rounds=5)
            paired.append((ratio, candidate_us, incumbent_us, variant))
            print("v195 qkv32 pair", variant, "candidate", candidate_us,
                  "incumbent", incumbent_us, "core_ratio", ratio, flush=True)
        if paired:
            ratio, candidate_us, incumbent_us, variant = min(paired)
            keep = candidate_us < incumbent_us and ratio <= _V195_ROUTE_RATIO
            if keep:
                state["v195_enabled"] = True
                state["v195_plan"] = plans[variant]
            print("v195 qkv32 decision", variant, "core_ratio", ratio,
                  "qualified", keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state["v195_enabled"] = False
        state["v195_plan"] = None
        print("v195 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["m"] == 32 and cfg["layer"] == "qkv_gate" and
            state.get("v195_enabled") and state.get("v195_plan") is not None):
        return _v158_call(state["v150"], state["v195_plan"], data["x"])
    return _v195_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
