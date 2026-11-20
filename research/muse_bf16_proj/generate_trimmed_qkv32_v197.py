"""Generate v197: v192 plus the single proven QKV/m32 RS schedule.

The v195 tournament selected stage-5 with the 9437219 policy at a 0.9715
paired ratio.  Keep only that template instantiation and enable it after the
existing bitwise validator succeeds, eliminating six unused variants and the
run-time timing tournament.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_relaxed_micro_composite_v192.py"
OUTPUT = ROOT / "submission_plain_trimmed_qkv32_v197.py"

text = BASE.read_text(encoding="utf-8")

old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v192 rq switch anchor count")
text = text.replace(old_switch, new_switch, 1)

block = r'''

# ---------------------------------------------------- v197: proven QKV/m32 route.
_v197_base_prepare = prepare
_v197_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v197_base_prepare(weights)
    state["v197_qkv32_plan"] = None
    vstate = state.get("v150")
    source = None if vstate is None else vstate.get("plans", {}).get("qkv_gate")
    if source is None:
        return state
    try:
        plan = dict(source)
        plan["variant"] = 2
        if _v158_validate(vstate, plan, weights["qkv_gate"], "qkv_gate", 32):
            state["v197_qkv32_plan"] = plan
            print("v197 qkv32 enabled stage5 policy9437219", flush=True)
        else:
            print("v197 qkv32 validator rejected", flush=True)
    except Exception as exc:
        state["v197_qkv32_plan"] = None
        print("v197 qkv32 unavailable", type(exc).__name__, str(exc)[:240],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    plan = state.get("v197_qkv32_plan")
    if cfg["m"] == 32 and cfg["layer"] == "qkv_gate" and plan is not None:
        return _v158_call(state["v150"], plan, data["x"])
    return _v197_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
