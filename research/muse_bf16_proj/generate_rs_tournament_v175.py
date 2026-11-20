"""Generate v175: bounded exact RS16/RS32 schedule tournament.

The current RS32 QKV winner is retained.  For the remaining high-leverage
pooler/unpooler routes (and QKV16), compile only upstream exact cursor/half
schedule families, screen them under cold-L2 graph replay, validate the three
fastest, and enable a route only at an overhead-corrected >=3% win.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zlib

import generate_rs128_faithful_v150 as v150
import generate_rs_small_v158 as v158


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_clean_composite_v174.py"
OUTPUT = ROOT / "submission_rs_tournament_v175.py"

# Keep the first five indices byte-for-byte compatible with v158/v174.
variants = (
    ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939),
    ("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
    ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075),
    ("t16s5hzw", 16, 5, 1, 0, 0, 0, 9437187),
    ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
    ("t32s5hzw", 32, 5, 1, 0, 0, 0, 9437187),
    ("t32s6hzwE", 32, 6, 1, 0, 0, 0, 9437219),
    ("t32s5hzwE", 32, 5, 1, 0, 0, 0, 9437219),
    ("t32s6hze", 32, 6, 1, 0, 0, 0, 35),
    ("t32s5hze", 32, 5, 1, 0, 0, 0, 35),
    ("t16s4hzw", 16, 4, 1, 0, 0, 0, 9437187),
    ("h16s5hzw2", 16, 5, 1, 0, 0, 0, 9438227),
    ("h16s5hze2", 16, 5, 1, 0, 0, 0, 1075),
    ("t16s4hze", 16, 4, 1, 0, 0, 0, 35),
)

cases = []
for index, (_, nt, st, gs, prb, wm, ins, fl) in enumerate(variants):
    cases.append(
        f"    case {index}: return rq_go<{nt}, {st}, {gs}, {prb}, {wm}, {ins}, {fl}, 1>"
        "(p, x, out, (int)KS, (int)info);"
    )

cuda_source = v150.rqcu.replace(
    "@@WGMMA@@",
    v150.wg_namespace["_rqwg"](16)
    + v150.wg_namespace["_rqwg"](32)
    + v150.wg_namespace["_rqwg"](128)
    + v150.wg_namespace["_rqwg"](256)
    + v150.wg_namespace["_rqss"](128)
    + v150.wg_namespace["_rqss"](256),
).replace("@@CASES@@", "\n".join(cases))
cuda_source = cuda_source.replace(
    "mbarrier.arrive.relaxed.cta.shared::cta.b64",
    "mbarrier.arrive.shared::cta.b64",
)
cuda_payload = base64.b64encode(zlib.compress(cuda_source.encode(), 9)).decode()

text = BASE.read_text()
if text.count(v158.cuda_payload) != 1:
    raise RuntimeError("unexpected v174 CUDA payload")
text = text.replace(v158.cuda_payload, cuda_payload, 1)

# The larger exact-template set remains under the 300 s prepare contract, but
# allow its background build more of that contract than the five-variant build.
if text.count("220.0 - (_v150_clock.time() - started)") != 1:
    raise RuntimeError("unexpected v174 build timeout")
text = text.replace(
    "220.0 - (_v150_clock.time() - started)",
    "265.0 - (_v150_clock.time() - started)",
    1,
)

text += r'''


# ------------------------------------------------ v175: exact RS16/RS32 schedule tournament.
_V175_ROUTES = {
    (16, "qkv_gate"): (3, 4, 10, 11, 12, 13),
    (16, "pooler"): (3, 4, 10, 11, 12, 13),
    (16, "unpooler"): (3, 4, 10, 11, 12, 13),
    (32, "pooler"): (1, 2, 5, 6, 7, 8, 9),
    (32, "unpooler"): (1, 2, 5, 6, 7, 8, 9),
}
_V175_ROUTE_RATIO = 0.97
_v175_base_prepare = prepare
_v175_base_kernel_fn = kernel_fn


def _v175_source_plan(state, weights, layer):
    vstate = state.get("v150")
    if vstate is None:
        return None
    source = vstate["plans"].get(layer)
    if source is not None:
        return source
    weight = weights[layer]
    packed = _v150_pack(weight)
    if packed is None:
        return None
    lay, tab, esc6 = packed
    if not torch.equal(
        _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
        weight.view(torch.int16).int() & 0xffff,
    ):
        return None
    ks = _V158_KS[layer]
    esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, weight.device)
    esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, weight.device)
    if esc2 is None:
        return None
    return {
        "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
        "esc2_max": esc2_max,
        "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=weight.device),
        "KS": ks, "n": weight.shape[0],
    }


def prepare(weights):
    state = _v175_base_prepare(weights)
    state["v175_enabled"] = set()
    state["v175_plans"] = {}
    vstate = state.get("v150")
    if vstate is None or not state.get("v150_enabled"):
        return state

    dev = weights["pooler"].device
    try:
        props = torch.cuda.get_device_properties(dev)
        if props.major != 9 or "H200" not in props.name:
            return state
        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(175032)

        for (m, layer), route_variants in _V175_ROUTES.items():
            source = _v175_source_plan(state, weights, layer)
            if source is None:
                continue
            weight = weights[layer]
            x = torch.randn((m, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": m, "n": weight.shape[0], "k": weight.shape[1]}
            data = {"x": x, "config": cfg}
            incumbent = lambda data=data: _v175_base_kernel_fn(data, state)
            incumbent_screen = _v150_time(incumbent, flush, repeats=9)

            screened = []
            plans = {}
            outputs = {}
            for variant in route_variants:
                plan = dict(source)
                plan["variant"] = variant
                plans[variant] = plan
                outputs[variant] = torch.empty(
                    (m, weight.shape[0]), dtype=torch.bfloat16, device=dev)
                fn = lambda plan=plan, out=outputs[variant], x=x: _v158_call(vstate, plan, x, out)
                samples = [_v150_time(fn, flush, repeats=7) for _ in range(2)]
                samples.sort()
                screened.append((samples[len(samples) // 2], variant))
            screened.sort()
            print("v175 screen", layer, m, "incumbent", incumbent_screen,
                  "candidates", screened, flush=True)

            paired = []
            for _, variant in screened[:3]:
                plan = plans[variant]
                if not _v158_validate(vstate, plan, weight, layer, m):
                    continue
                out = outputs[variant]
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda plan=plan, out=out, x=x: _v158_call(vstate, plan, x, out),
                    incumbent, flush, repeats=11, rounds=3,
                )
                paired.append((ratio, candidate_us, incumbent_us, variant))
                print("v175 pair", layer, m, "variant", variant,
                      "candidate", candidate_us, "incumbent", incumbent_us,
                      "core_ratio", ratio, flush=True)

            if paired:
                ratio, candidate_us, incumbent_us, variant = min(paired)
                keep = candidate_us < incumbent_us and ratio <= _V175_ROUTE_RATIO
                if keep:
                    state["v175_enabled"].add((m, layer))
                    state["v175_plans"][(m, layer)] = plans[variant]
                print("v175 decision", layer, m, "variant", variant,
                      "core_ratio", ratio, "qualified", keep, flush=True)

        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v175 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v175_enabled", ()):
        plan = state.get("v175_plans", {}).get(key)
        if plan is not None:
            return _v158_call(state["v150"], plan, data["x"])
    return _v175_base_kernel_fn(data, state)
'''

compile(text, str(OUTPUT), "exec")

if __name__ == "__main__":
    OUTPUT.write_text(text)
    print(OUTPUT)
    print("variants", variants)
    print("cuda_sha256", hashlib.sha256(cuda_source.encode()).hexdigest().upper())
