"""Generate v174: clean QKV composite plus guarded small-QKV routes.

Start from v170's validated QKV32 and exact QKV256 composition, remove the
unstable router-m32 experiment, then race the embedded alternate implementation
only at QKV m=1 and m=8.  Those two routes are enabled independently only after
exact one-hot, FP64-reference, and paired cold-L2 validation.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_composite_v170.py"
OUTPUT = ROOT / "submission_clean_composite_v174.py"


text = BASE.read_text()

router_patch = '''_ALT_DECODED = _zl.decompress(_b64.b64decode(_ALT_SRC)).decode()
_R32_BASE = "ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]"
_R32_ST8 = "ROUTER_CONFIGS = [(32,16,8,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]"
if _ALT_DECODED.count(_R32_BASE) != 1:
    raise RuntimeError("Router m32 baseline config was not found exactly once")
_ALT_DECODED = _ALT_DECODED.replace(_R32_BASE, _R32_ST8)
exec(compile(_ALT_DECODED, "_alt_proj", "exec"), _alt.__dict__)'''
clean_loader = '''exec(compile(_zl.decompress(_b64.b64decode(_ALT_SRC)).decode(), "_alt_proj", "exec"), _alt.__dict__)'''
if text.count(router_patch) != 1:
    raise RuntimeError("unexpected v170 router patch")
text = text.replace(router_patch, clean_loader, 1)

text += r'''


# --------------------------------------------------------- v174: guarded QKV m1/m8 alternate routes.
_V174_ALT_TARGETS = (1, 8)
_V174_ROUTE_RATIO = 0.97
_v174_base_prepare = prepare
_v174_base_kernel_fn = kernel_fn


def _v174_alt_call(data, state):
    return _alt.kernel_fn(data, state["_alt"])


def _v174_validate_alt(state, weight, m):
    n, k = weight.shape
    cfg = {"layer": "qkv_gate", "m": m, "n": n, "k": k}
    cols = (0, k // 3, (2 * k) // 3, k - 1)

    exact = True
    for col in cols:
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
        x[0, col] = 8.0
        out = _v174_alt_call({"x": x, "config": cfg}, state)
        torch.cuda.synchronize()
        want = (weight[:, col] * 8.0).to(torch.bfloat16)
        exact = exact and torch.equal(out[0], want)
        if m > 1:
            exact = exact and bool((out[1] == 0).all())

    gen = torch.Generator(device=weight.device).manual_seed(174000 + m)
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v174_alt_call({"x": x, "config": cfg}, state)
    torch.cuda.synchronize()
    ref = x.double() @ weight.double().T
    diff = out.double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    ok = exact and bool(torch.isfinite(out).all()) and rel <= 0.01 and worst <= 0.05
    print("v174 alt qkv", m, "one_hot", exact, "fp64", rel, worst,
          "valid", ok, flush=True)
    return ok


def prepare(weights):
    state = _v174_base_prepare(weights)
    state["v174_alt_enabled"] = set()
    if "_alt" not in state:
        return state

    dev = weights["qkv_gate"].device
    try:
        props = torch.cuda.get_device_properties(dev)
        if props.major != 9 or "H200" not in props.name:
            return state
        weight = weights["qkv_gate"]
        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(174108)
        for m in _V174_ALT_TARGETS:
            if not _v174_validate_alt(state, weight, m):
                continue
            x = torch.randn((m, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": "qkv_gate", "m": m,
                   "n": weight.shape[0], "k": weight.shape[1]}
            data = {"x": x, "config": cfg}
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda data=data: _v174_alt_call(data, state),
                lambda data=data: _v174_base_kernel_fn(data, state),
                flush, repeats=13, rounds=3,
            )
            keep = candidate_us < incumbent_us and ratio <= _V174_ROUTE_RATIO
            if keep:
                state["v174_alt_enabled"].add((m, "qkv_gate"))
            print("v174 alt qkv", m, "candidate", candidate_us,
                  "incumbent", incumbent_us, "core_ratio", ratio,
                  "qualified", keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v174 alt unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["m"], cfg["layer"]) in state.get("v174_alt_enabled", ()):
        return _v174_alt_call(data, state)
    return _v174_base_kernel_fn(data, state)
'''

OUTPUT.write_text(text)
print(OUTPUT)
