"""Generate v172: v170 plus guarded simple-cursor RS128 and Nathen O16."""

from pathlib import Path
import base64
import hashlib
import zlib

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_composite_v170.py"
OUTPUT = ROOT / "submission_best_composite_v172.py"


# Preserve every CUDA case index used by v158/v170 and append the two upstream
# RS128 simple-cursor schedules.  Indices 5 and 6 are new in this candidate.
variants = (
    (128, 5, 1, 0, 0, 0, 42999939),
    (32, 6, 1, 0, 0, 0, 9437187),
    (32, 5, 1, 0, 0, 0, 1075),
    (16, 5, 1, 0, 0, 0, 9437187),
    (16, 5, 1, 0, 0, 0, 35),
    (128, 5, 1, 0, 0, 0, 9437187),   # s4s5hzw
    (128, 5, 1, 0, 0, 0, 42991619),  # s4s5hzwP
)
cases = "\n".join(
    f"    case {i}: return rq_go<{nt}, {st}, {gs}, {prb}, {wm}, {ins}, {fl}, 1>"
    "(p, x, out, (int)KS, (int)info);"
    for i, (nt, st, gs, prb, wm, ins, fl) in enumerate(variants)
)
cuda = v150.rqcu.replace(
    "@@WGMMA@@",
    v150.wg_namespace["_rqwg"](16)
    + v150.wg_namespace["_rqwg"](32)
    + v150.wg_namespace["_rqwg"](128)
    + v150.wg_namespace["_rqwg"](256)
    + v150.wg_namespace["_rqss"](128)
    + v150.wg_namespace["_rqss"](256),
).replace("@@CASES@@", cases)
cuda = cuda.replace(
    "mbarrier.arrive.relaxed.cta.shared::cta.b64",
    "mbarrier.arrive.shared::cta.b64",
)
payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()

text = BASE.read_text()
start = '_V150_CUDA_PAYLOAD = "'
a = text.index(start) + len(start)
b = text.index('"\n_V150_VARIANT', a)
text = text[:a] + payload + text[b:]

# The embedded Nathen module already builds and validates its O16 route.  It
# beat the current O16 path by about 0.46 us in its full H200 run, so expose
# only that one additional shape.
old_routes = "_ALT_ROUTES = frozenset([('o', 8), ('o', 128), ('o', 256),"
new_routes = "_ALT_ROUTES = frozenset([('o', 8), ('o', 16), ('o', 128), ('o', 256),"
if text.count(old_routes) != 1:
    raise RuntimeError("unexpected alternate-route table")
text = text.replace(old_routes, new_routes, 1)

append = r'''

# ---------------------------------------------------------------- v172: upstream RS128 simple-cursor tournament.
_V172_TARGETS = ("pooler", "unpooler")
_V172_VARIANTS = (5, 6)
_V172_ROUTE_RATIO = 0.97


def _v172_call(vstate, plan, variant, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = vstate["ext"].rqrun(
        variant, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], info,
    )
    if rc:
        raise RuntimeError("v172 rqrun variant %d error %d" % (variant, rc))
    return out


def _v172_info(vstate, plan, variant, kind):
    eb = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    ei = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(vstate["ext"].rqrun(
        variant, eb, eb, eb, ei, ei, ei, plan["KS"], plan["tab"], 0,
        vstate["prof"], vstate["sink"], kind,
    ))


def _v172_validate(vstate, plan, weight, layer, variant):
    m, k, ks = 128, weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    plan["cnt"].zero_()
    out = _v172_call(vstate, plan, variant, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())

    gen = torch.Generator(device=weight.device).manual_seed(172000 + variant + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    x[:, torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v172_call(vstate, plan, variant, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    clean = int(plan["cnt"].abs().sum()) == 0
    if not clean:
        plan["cnt"].zero_()
    ok = exact and clean and bool(torch.isfinite(out).all()) and rel <= _V150_TOL[ks] and worst <= 5 * _V150_TOL[ks]
    print("v172 exact", layer, "variant", variant, "one_hot", exact,
          "fp64", rel, worst, "regs", _v172_info(vstate, plan, variant, 1),
          "local", _v172_info(vstate, plan, variant, 2),
          "smem", _v172_info(vstate, plan, variant, 3), "valid", ok, flush=True)
    return ok


_v172_base_prepare = prepare
_v172_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v172_base_prepare(weights)
    state["v172_routes"] = {}
    vstate = state.get("v150")
    if vstate is None or not state.get("v150_enabled"):
        return state
    try:
        dev = weights["pooler"].device
        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(172128)
        for layer in _V172_TARGETS:
            plan = vstate["plans"].get(layer)
            if plan is None:
                continue
            valid = [v for v in _V172_VARIANTS if _v172_validate(vstate, plan, weights[layer], layer, v)]
            if not valid:
                continue
            x = torch.randn((128, weights[layer].shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            incumbent_out = torch.empty((128, weights[layer].shape[0]), dtype=torch.bfloat16, device=dev)
            results = []
            for variant in valid:
                candidate_out = torch.empty_like(incumbent_out)
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda variant=variant, candidate_out=candidate_out: _v172_call(vstate, plan, variant, x, candidate_out),
                    lambda: _v150_call(vstate, plan, x, incumbent_out), flush,
                )
                keep = candidate_us < incumbent_us and ratio <= _V172_ROUTE_RATIO
                results.append((ratio, candidate_us, variant, keep))
                print("v172 rs128", layer, "variant", variant, "candidate", candidate_us,
                      "incumbent", incumbent_us, "core_ratio", ratio,
                      "qualified", keep, flush=True)
            winners = [item for item in results if item[3]]
            if winners:
                state["v172_routes"][layer] = min(winners)[2]
        print("v172 enabled", state["v172_routes"], flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v172 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if cfg["m"] == 128:
        variant = state.get("v172_routes", {}).get(cfg["layer"])
        if variant is not None:
            plan = state["v150"]["plans"].get(cfg["layer"])
            if plan is not None:
                return _v172_call(state["v150"], plan, variant, data["x"])
    return _v172_base_kernel_fn(data, state)
'''

text += append
compile(text, str(OUTPUT), "exec")
OUTPUT.write_text(text)
print(OUTPUT)
print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
print("candidate_sha256", hashlib.sha256(text.encode()).hexdigest().upper())
