"""Generate v158: protected RS128 plus guarded RS16/RS32 tournaments.

The Apache-2.0 upstream source contains judge-proven cursor variants for the
exact m=16 and m=32 shapes, but v150 intentionally ported only its RS128 path.
This candidate compiles both families in one extension, validates every small
route, and enables them only when their combined cold-L2 projection clears the
same whole-score gate used by v150.  Every non-winning route remains v150/v105.
"""

from __future__ import annotations

import base64
import hashlib
import zlib

import generate_rs128_faithful_v150 as v150


OUT = v150.ROOT / "submission_rs_small_v158.py"

variants = (
    ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939),
    ("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
    ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075),
    ("t16s5hzw", 16, 5, 1, 0, 0, 0, 9437187),
    ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
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
prelude = v150.prelude.replace(v150.cuda_payload, cuda_payload)

integration = r'''

# ---------------------------------------------------------------- v158: exact RS16/RS32 guarded tournament.
_V158_ROUTES = {
    (32, "qkv_gate"): 1,
    (32, "pooler"): 1,
    (32, "unpooler"): 1,
    (32, "o"): 2,
    (16, "qkv_gate"): 3,
    (16, "pooler"): 4,
    (16, "unpooler"): 4,
    (16, "o"): 4,
}
_V158_KS = {"qkv_gate": 7, "pooler": 4, "unpooler": 2, "o": 2}
_V158_ROUTE_RATIO = 0.97
_V158_SCORE_GATE = 0.009


def _v158_call(vstate, plan, x, out=None):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = vstate["ext"].rqrun(
        plan["variant"], plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], 0,
    )
    if rc != 0:
        raise RuntimeError("v158 rqrun error %d" % rc)
    return out


def _v158_info(vstate, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(vstate["ext"].rqrun(
        plan["variant"], empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], kind,
    ))


def _v158_validate(vstate, plan, weight, layer, m):
    n, k, ks = weight.shape[0], weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v158_call(vstate, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want)
    if len(cols) < m:
        exact = exact and bool((out[len(cols)] == 0).all())

    gen = torch.Generator(device=weight.device).manual_seed(158000 + m + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v158_call(vstate, plan, x)
    torch.cuda.synchronize()
    ref = x[:min(8, m)].double() @ weight.double().T
    diff = out[:min(8, m)].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL[ks]
    counters_ok = int(plan["cnt"].abs().sum()) == 0
    if not counters_ok:
        plan["cnt"].zero_()
    ok = exact and bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol and counters_ok
    print(
        "v158 exact", layer, "m", m, "variant", plan["variant"],
        "one_hot", exact, "fp64", rel, worst,
        "regs", _v158_info(vstate, plan, 1), "local", _v158_info(vstate, plan, 2),
        "smem", _v158_info(vstate, plan, 3), "occupancy", _v158_info(vstate, plan, 4),
        "valid", ok, flush=True,
    )
    return ok


_v150_prepare_with_rs128 = prepare
_v150_kernel_with_rs128 = kernel_fn


def prepare(weights):
    state = _v150_prepare_with_rs128(weights)
    state["v158_enabled"] = set()
    state["v158_projected_gain"] = 0.0
    vstate = state.get("v150")
    if vstate is None or not state.get("v150_enabled"):
        return state

    dev = weights["pooler"].device
    try:
        plans = {}
        for (m, layer), variant in _V158_ROUTES.items():
            source_plan = vstate["plans"].get(layer)
            if source_plan is None:
                weight = weights[layer]
                packed = _v150_pack(weight)
                if packed is None:
                    continue
                lay, tab, esc6 = packed
                if not torch.equal(
                    _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                    weight.view(torch.int16).int() & 0xffff,
                ):
                    continue
                ks = _V158_KS[layer]
                esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
                esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
                if esc2 is None:
                    continue
                source_plan = {
                    "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                    "esc2_max": esc2_max,
                    "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                    "KS": ks, "n": weight.shape[0],
                }
            plan = dict(source_plan)
            plan["variant"] = variant
            if _v158_validate(vstate, plan, weights[layer], layer, m):
                plans[(m, layer)] = plan
        state["v158_plans"] = plans
        if not plans:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(158032)
        qualified = set()
        ratios_by_m = {}
        for m in (16, 32):
            inc_total = cand_total = 0.0
            for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
                weight = weights[layer]
                x = torch.randn((m, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
                cfg = {"layer": layer, "m": m, "n": weight.shape[0], "k": weight.shape[1]}
                plan = plans.get((m, layer))
                if plan is None:
                    unchanged = _v150_time(
                        lambda x=x, cfg=cfg: _v150_kernel_with_rs128({"x": x, "config": cfg}, state), flush,
                    )
                    core = max(unchanged - _V150_REPLAY_US, 0.05)
                    inc_total += core; cand_total += core
                    continue
                out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=dev)
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda x=x, out=out, plan=plan: _v158_call(vstate, plan, x, out),
                    lambda x=x, cfg=cfg: _v150_kernel_with_rs128({"x": x, "config": cfg}, state),
                    flush,
                )
                keep = candidate_us < incumbent_us and ratio <= _V158_ROUTE_RATIO
                if keep:
                    qualified.add((m, layer))
                inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
                cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
                inc_total += inc_core; cand_total += cand_core
                print(
                    "v158 rs-small", layer, "m", m, "variant", plan["variant"],
                    "candidate", candidate_us, "incumbent", incumbent_us,
                    "core_ratio", ratio, "qualified", keep, flush=True,
                )
            ratios_by_m[m] = inc_total / cand_total
            print("v158 load", m, "incumbent", inc_total, "candidate", cand_total,
                  "ratio", ratios_by_m[m], flush=True)

        projected = (ratios_by_m[16] * ratios_by_m[32]) ** (1.0 / 6.0) - 1.0
        state["v158_projected_gain"] = projected
        if projected >= _V158_SCORE_GATE:
            state["v158_enabled"] = qualified
        print(
            "v158 projected_score_gain", projected,
            "enabled", sorted(state["v158_enabled"]), flush=True,
        )
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v158 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v158_enabled", ()):
        plan = state.get("v158_plans", {}).get(key)
        if plan is not None:
            return _v158_call(state["v150"], plan, data["x"])
    return _v150_kernel_with_rs128(data, state)
'''

candidate = (
    v150.base
    + prelude
    + "\n\n"
    + v150.pack_source
    + "\n\n"
    + v150.integration.lstrip()
    + "\n\n"
    + integration.lstrip()
)
compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("variants", variants)
    print("cuda_sha256", hashlib.sha256(cuda_source.encode()).hexdigest().upper())
    print("baseline_prefix", candidate.encode()[: len(v150.base_bytes)] == v150.base_bytes)
    print("bytes", len(candidate.encode()))
