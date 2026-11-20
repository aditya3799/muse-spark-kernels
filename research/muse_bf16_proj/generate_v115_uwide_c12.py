"""Generate v115 from byte-identical v105 plus a gated m=256 C12/WGMMA path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_cgemv_threads_v105.py"
V55 = ROOT / "submission_unpooler128_c12_v55.py"
OUT = ROOT / "submission_uwide_c12_m256_v115.py"

base_bytes = BASE.read_bytes()
base = base_bytes.decode()
old = V55.read_text()

start = old.index("# ---------------------------------------------------------------- v32: lossless C12 unpooler-32")
end = old.index("\n_u32_mod = [None]", start)
section = old[start:end]
section = section.replace("_q128_zsrc(", "_v115_zsrc(")
section = section.replace(
    "ext = _u32_ext() if instances == _u32_INSTANCES else _u32_build(instances)",
    "ext = _u32_build(instances)",
)
section = section.replace(
    "_u32_CUDA_CORE + _u32_CUDA_MAP",
    "_v115_core() + _u32_CUDA_MAP",
)
section = section.replace(
    "(_u32_mma(n) for n in (16, 32, 64))",
    "(_u32_mma(n) for n in (16, 32, 64, 128, 256))",
)
section = section.replace("for c in (inst[5], False):", "for c in (inst[5],):")
section = section.replace(
    "for compressed in (inst[5], False):", "for compressed in (inst[5],):"
)

prelude = r'''
import base64 as _v115_base64
import hashlib
import zlib as _v115_zlib


def _v115_zsrc(payload):
    return _v115_zlib.decompress(_v115_base64.b64decode(payload)).decode()
'''

integration = r'''

# ---------------------------------------------------------------- v115: broad m=256 lossless-C12 full-token WGMMA.
# The inherited v55 kernel decoded one C12 fragment for only 64 tokens and
# launched two CTA waves at m=128.  This specialization uses m64n256 WGMMA:
# every exact decoded weight fragment feeds all 256 tokens.  Shape-specific
# cluster split-K keeps the grid close to one H200 wave without changing BF16
# inputs/weights, FP32 accumulation, or the final BF16 conversion.
import base64 as _v115_base64
import hashlib as _v115_hashlib
import zlib as _v115_zlib


def _v115_zsrc(payload):
    return _v115_zlib.decompress(_v115_base64.b64decode(payload)).decode()


def _v115_core():
    return _u32_CUDA_CORE.replace(
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));",
        "if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n"
        "      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));",
    )


_V115_INST = (256, 64, 2, 2, 2, True, 1)
_V115_CONFIGS = {
    (256, "qkv_gate"): (_V115_INST, 4),
    (256, "o"): (_V115_INST, 1),
    (256, "pooler"): (_V115_INST, 2),
    (256, "unpooler"): (_V115_INST, 1),
}
_V115_REPLAY_US = 4.0
_V115_ROUTE_RATIO = 0.97
_V115_SCORE_GATE = 0.02
_V115_FLUSH = [None, None]


def _v115_time(fn, flush, repeats=13):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fn()
    if _V115_FLUSH[0] is not flush:
        _V115_FLUSH[0] = flush
        _V115_FLUSH[1] = torch.empty(1024, dtype=torch.float32, device=flush.device)
    samples = []
    for _ in range(repeats):
        torch.sum(flush, dim=0, out=_V115_FLUSH[1])
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _v115_pair(candidate, incumbent, flush, repeats=13, rounds=3):
    ratios, cand, inc = [], [], []
    for r in range(rounds):
        if r & 1:
            b = _v115_time(incumbent, flush, repeats)
            a = _v115_time(candidate, flush, repeats)
        else:
            a = _v115_time(candidate, flush, repeats)
            b = _v115_time(incumbent, flush, repeats)
        a_core = max(a * 1000.0 - _V115_REPLAY_US, 0.05)
        b_core = max(b * 1000.0 - _V115_REPLAY_US, 0.05)
        ratios.append(a_core / b_core)
        cand.append(a * 1000.0)
        inc.append(b * 1000.0)
    ratios.sort()
    cand.sort()
    inc.sort()
    return ratios[len(ratios) // 2], cand[len(cand) // 2], inc[len(inc) // 2]


def _v115_validate(plan_state, weights):
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        plan = plan_state.get("plans", {}).get((256, layer))
        if plan is None or not plan[1].get("compressed", False):
            print("v115 rejected", layer, "missing exact C12 plan", flush=True)
            return False
        weight = weights[layer]
        gen = torch.Generator(device=weight.device).manual_seed(1150 + len(layer))
        x = torch.zeros((256, weight.shape[1]), dtype=torch.bfloat16, device=weight.device)
        x[:8].copy_(torch.randn((8, weight.shape[1]), generator=gen,
                                dtype=torch.float32, device=weight.device).to(torch.bfloat16))
        x[7].zero_()
        x[7, weight.shape[1] - 17] = 16
        out = _u32_run_m32w(x, layer, weight.shape[0], plan_state)
        if out is None or not bool(torch.isfinite(out).all()):
            print("v115 rejected", layer, "nonfinite or missing output", flush=True)
            return False
        ref = x[:8].double() @ weight.double().T
        diff = out[:8].double() - ref
        rel = float(diff.norm() / ref.norm())
        den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        worst = float((diff.norm(dim=1) / den).max())
        if not (rel < 0.005 and worst < 0.01 and bool((out[8:] == 0).all())):
            print("v115 rejected", layer, "fp64", rel, worst, flush=True)
            return False
        print("v115 exact", layer, "fp64", rel, worst,
              "escapes", plan[1].get("escapes", -1),
              "max", plan[1].get("max_esc", -1), flush=True)
    return True


_v105_prepare = prepare
_v105_kernel_fn = kernel_fn


def prepare(weights):
    state = _v105_prepare(weights)
    state["v115_enabled"] = set()
    state["v115_projected_gain"] = 0.0
    dev = weights["o"].device
    props = torch.cuda.get_device_properties(dev)
    if props.major != 9 or "H200" not in props.name:
        return state

    plan_state = _u32_prepare_m32w(
        {name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler")},
        configs=_V115_CONFIGS,
    )
    state["v115"] = plan_state
    if not _v115_validate(plan_state, weights):
        return state

    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024,
                       dtype=torch.float32, device=dev)
    route_times = {}
    qualified = set()
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        weight = weights[layer]
        x = torch.randn((256, weight.shape[1]), device=dev).to(torch.bfloat16)
        cfg = {"layer": layer, "m": 256, "n": weight.shape[0], "k": weight.shape[1]}
        ratio, cand_us, inc_us = _v115_pair(
            lambda layer=layer, x=x, n=weight.shape[0]: _u32_run_m32w(x, layer, n, plan_state),
            lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
            flush,
        )
        keep = cand_us < inc_us and ratio <= _V115_ROUTE_RATIO
        if keep:
            qualified.add(layer)
        route_times[layer] = (cand_us, inc_us, ratio, keep)
        print("v115 m256", layer, "candidate", cand_us, "incumbent", inc_us,
              "core_ratio", ratio, "qualified", keep, flush=True)

    rw = weights["router"]
    rx = torch.randn((256, rw.shape[1]), device=dev).to(torch.bfloat16)
    rcfg = {"layer": "router", "m": 256, "n": rw.shape[0], "k": rw.shape[1]}
    router_us = _v115_time(lambda: _v105_kernel_fn({"x": rx, "config": rcfg}, state), flush)
    router_core = max(router_us * 1000.0 - _V115_REPLAY_US, 0.05)
    incumbent_load = router_core
    candidate_load = router_core
    for layer, (cand_us, inc_us, _, keep) in route_times.items():
        inc_core = max(inc_us - _V115_REPLAY_US, 0.05)
        cand_core = max(cand_us - _V115_REPLAY_US, 0.05)
        incumbent_load += inc_core
        candidate_load += cand_core if keep else inc_core
    projected = (incumbent_load / candidate_load) ** (1.0 / 6.0) - 1.0
    state["v115_projected_gain"] = projected
    if projected >= _V115_SCORE_GATE:
        state["v115_enabled"] = qualified
    print("v115 m256 load incumbent", incumbent_load, "candidate", candidate_load,
          "projected_score_gain", projected,
          "enabled", sorted(state["v115_enabled"]), flush=True)
    del flush
    torch.cuda.empty_cache()
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    layer = cfg["layer"]
    if cfg["m"] == 256 and layer in state.get("v115_enabled", ()):
        out = _u32_run_m32w(data["x"], layer, cfg["n"], state["v115"])
        if out is not None:
            return out
    return _v105_kernel_fn(data, state)
'''

candidate = base + "\n" + prelude.lstrip() + "\n" + section + "\n\n" + integration.lstrip() + "\n"
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("baseline_prefix", candidate.encode()[:len(base_bytes)] == base_bytes)
print("bytes", len(candidate.encode()))
