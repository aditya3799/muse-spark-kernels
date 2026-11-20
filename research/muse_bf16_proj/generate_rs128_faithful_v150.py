"""Generate v150: faithful chenwainuo RS128 for three m=128 projections.

The protected v105 submission is retained byte-for-byte as the candidate prefix.
Only the Apache-2.0 RS128 CUDA translation unit and its exact C12 packer are
ported from chenwainuo/okc at commit 3d418e1.  The generated candidate bakes the
single upstream-proven d2s5hzwP path against v105 for qkv_gate, pooler, and
unpooler at m=128; every other route remains v105.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import tempfile
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_cgemv_threads_v105.py"
OUT = ROOT / "submission_rs128_faithful_v150.py"
UPSTREAM_CACHE = Path(tempfile.gettempdir()) / "chenwainuo_3d418e1_submission.py"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/chenwainuo/okc/3d418e1/"
    "muse_bf16_proj/submission.py"
)
UPSTREAM_SHA256 = "BB50079BE2ED74D65BB7109F1432FBC7A49BBE70913D247C254717CD0F27F712"


def _upstream_text() -> str:
    if UPSTREAM_CACHE.exists():
        data = UPSTREAM_CACHE.read_bytes()
    else:
        with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError(f"upstream SHA256 mismatch: {digest}")
    return data.decode()


def _node_by_name(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    raise KeyError(name)


upstream = _upstream_text()
tree = ast.parse(upstream)

rqcu_node = _node_by_name(tree, "RQCU")
assert isinstance(rqcu_node, ast.Assign)
rqcu = ast.literal_eval(rqcu_node.value)

wg_namespace: dict[str, object] = {}
for name in ("_rqwg", "_rqss"):
    node = _node_by_name(tree, name)
    source = ast.get_source_segment(upstream, node)
    assert source is not None
    exec(compile(source, f"<upstream:{name}>", "exec"), wg_namespace)

variant = ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939)
case = "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);"
cuda_source = rqcu.replace(
    "@@WGMMA@@",
    wg_namespace["_rqwg"](16)
    + wg_namespace["_rqwg"](32)
    + wg_namespace["_rqwg"](128)
    + wg_namespace["_rqwg"](256)
    + wg_namespace["_rqss"](128)
    + wg_namespace["_rqss"](256),
).replace("@@CASES@@", case)
# CUDA 12.4 rejects the newer relaxed-arrive spelling even when the selected
# template has RLAX=false and dead-code elimination removes every call.  Keep
# the helper parseable by spelling it as the ordinary arrive used by
# d2s5hzwP; this does not change the instantiated path.
cuda_source = cuda_source.replace(
    "mbarrier.arrive.relaxed.cta.shared::cta.b64",
    "mbarrier.arrive.shared::cta.b64",
)
cuda_payload = base64.b64encode(zlib.compress(cuda_source.encode(), 9)).decode()

pack_nodes = [
    _node_by_name(tree, "_OS"),
    _node_by_name(tree, "_SS"),
    _node_by_name(tree, "_c12"),
    _node_by_name(tree, "_unc12"),
    _node_by_name(tree, "_pack_nat"),
    _node_by_name(tree, "_rqpack"),
    _node_by_name(tree, "_rqunpack"),
    _node_by_name(tree, "_rqesc"),
    _node_by_name(tree, "_rqesc2"),
]
pack_source = "\n\n".join(ast.get_source_segment(upstream, node) or "" for node in pack_nodes)
renames = {
    "_rqesc2": "_v150_esc2",
    "_rqunpack": "_v150_unpack",
    "_pack_nat": "_v150_pack_nat",
    "_rqpack": "_v150_pack",
    "_unc12": "_v150_unc12",
    "_rqesc": "_v150_esc",
    "_c12": "_v150_c12",
    "_OS": "_V150_OS",
    "_SS": "_V150_SS",
}
for old, new in renames.items():
    pack_source = pack_source.replace(old, new)

base_bytes = BASE.read_bytes()
base = base_bytes.decode()

prelude = f'''

# ---------------------------------------------------------------- v150: faithful RS128 producer/consumer pipeline.
# Apache-2.0 source: github.com/chenwainuo/okc @ 3d418e1.
# Upstream source SHA256: {UPSTREAM_SHA256}.
import base64 as _v150_base64
import time as _v150_clock
import hashlib as _v150_hashlib
import threading as _v150_threading
import zlib as _v150_zlib

_V150_CUDA_PAYLOAD = "{cuda_payload}"
_V150_VARIANT = {variant!r}
_V150_CPP = "int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);"


def _v150_cuda_source():
    return _v150_zlib.decompress(_v150_base64.b64decode(_V150_CUDA_PAYLOAD)).decode()


_v150_module = [None]


def _v150_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v150_rs128_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
        _v150_module[0] = load_inline(
            tag,
            cpp_sources=_V150_CPP,
            cuda_sources=src,
            functions=["rqrun"],
            extra_cuda_cflags=[
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3", "-DNDEBUG", "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _v150_module[0]
'''

integration = r'''

_V150_TARGETS = ("qkv_gate", "pooler", "unpooler")
_V150_KS = {"qkv_gate": 7, "pooler": 4, "unpooler": 2}
_V150_TOL = {2: 5e-3, 4: 6e-3, 7: 8e-3}
_V150_ROUTE_RATIO = 0.97
_V150_SCORE_GATE = 0.009
_V150_REPLAY_US = 4.0
_V150_FLUSH = [None, None]


def _v150_start_build():
    status = []

    def build():
        try:
            _v150_build()
            status.append("ok")
        except Exception as exc:
            status.append("failed: " + (str(exc).splitlines()[-1][:240] if str(exc) else type(exc).__name__))

    thread = _v150_threading.Thread(target=build, daemon=True)
    thread.start()
    return thread, status


def _v150_call(state, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        0, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v150 rqrun error %d" % rc)
    return out


def _v150_info(state, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        0, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v150_validate(state, plan, weight, layer):
    m, n, k, ks = 128, weight.shape[0], weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    out = _v150_call(state, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    if not exact or int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        print("v150 rejected", layer, "one-hot/counter", exact, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(1500 + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v150_call(state, plan, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL[ks]
    ok = bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol
    if int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        ok = False
    print(
        "v150 exact", layer, "one_hot", exact, "fp64", rel, worst,
        "regs", _v150_info(state, plan, 1), "local", _v150_info(state, plan, 2),
        "smem", _v150_info(state, plan, 3), "esc2_max", plan["esc2_max"],
        "valid", ok, flush=True,
    )
    return ok


def _v150_time(fn, flush, repeats=11):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fn()
    if _V150_FLUSH[0] is not flush:
        _V150_FLUSH[0] = flush
        _V150_FLUSH[1] = torch.empty(1024, dtype=torch.float32, device=flush.device)
    samples = []
    for _ in range(repeats):
        torch.sum(flush, dim=0, out=_V150_FLUSH[1])
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _v150_pair(candidate, incumbent, flush, repeats=11, rounds=3):
    ratios, cand, inc = [], [], []
    for round_index in range(rounds):
        if round_index & 1:
            b = _v150_time(incumbent, flush, repeats)
            a = _v150_time(candidate, flush, repeats)
        else:
            a = _v150_time(candidate, flush, repeats)
            b = _v150_time(incumbent, flush, repeats)
        ratios.append(
            max(a - _V150_REPLAY_US, 0.05) /
            max(b - _V150_REPLAY_US, 0.05)
        )
        cand.append(a); inc.append(b)
    ratios.sort(); cand.sort(); inc.sort()
    middle = len(ratios) // 2
    return ratios[middle], cand[middle], inc[middle]


_v105_prepare = prepare
_v105_kernel_fn = kernel_fn


def prepare(weights):
    started = _v150_clock.time()
    dev = weights["pooler"].device
    props = torch.cuda.get_device_properties(dev)
    eligible = props.major == 9 and "H200" in props.name
    build_thread = build_status = None
    if eligible:
        build_thread, build_status = _v150_start_build()

    state = _v105_prepare(weights)
    state["v150_enabled"] = set()
    state["v150_projected_gain"] = 0.0
    if not eligible:
        return state

    try:
        build_thread.join(timeout=max(1.0, 220.0 - (_v150_clock.time() - started)))
        if not build_status or build_status[0] != "ok":
            print("v150 unavailable build", build_status or "not finished", flush=True)
            return state
        vstate = {
            "ext": _v150_build(),
            "prof": torch.empty(0, dtype=torch.int64, device=dev),
            "sink": torch.zeros(4, dtype=torch.int32, device=dev),
            "plans": {},
        }
        state["v150"] = vstate

        for layer in _V150_TARGETS:
            weight = weights[layer]
            packed = _v150_pack(weight)
            if packed is None:
                print("v150 rejected", layer, "pack unavailable", flush=True)
                continue
            lay, tab, esc6 = packed
            if not torch.equal(
                _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                weight.view(torch.int16).int() & 0xffff,
            ):
                print("v150 rejected", layer, "pack mismatch", flush=True)
                continue
            ks = _V150_KS[layer]
            esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
            esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
            if esc2 is None:
                print("v150 rejected", layer, "esc2 overflow", esc2_max, flush=True)
                continue
            plan = {
                "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                "esc2_max": esc2_max, "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                "KS": ks, "n": weight.shape[0],
            }
            vstate["plans"][layer] = plan
            if not _v150_validate(vstate, plan, weight, layer):
                del vstate["plans"][layer]

        if not vstate["plans"]:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        qualified = set()
        route_times = {}
        inc_total = cand_total = 0.0
        gen = torch.Generator(device=dev).manual_seed(150128)
        for layer in _V150_TARGETS:
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 128, "n": weight.shape[0], "k": weight.shape[1]}
            plan = vstate["plans"].get(layer)
            if plan is None:
                incumbent_us = _v150_time(lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state), flush)
                inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
                inc_total += inc_core; cand_total += inc_core
                continue
            out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16, device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda x=x, out=out, plan=plan: _v150_call(vstate, plan, x, out),
                lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            keep = candidate_us < incumbent_us and ratio <= _V150_ROUTE_RATIO
            if keep:
                qualified.add((128, layer))
            route_times[layer] = (candidate_us, incumbent_us, ratio, keep)
            inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
            cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
            inc_total += inc_core; cand_total += cand_core
            print(
                "v150 rs128", layer, "candidate", candidate_us,
                "incumbent", incumbent_us, "core_ratio", ratio,
                "qualified", keep, flush=True,
            )

        for layer in ("o", "router"):
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 128, "n": weight.shape[0], "k": weight.shape[1]}
            unchanged_us = _v150_time(
                lambda x=x, cfg=cfg: _v105_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            unchanged_core = max(unchanged_us - _V150_REPLAY_US, 0.05)
            inc_total += unchanged_core; cand_total += unchanged_core

        load_ratio = inc_total / cand_total
        projected = load_ratio ** (1.0 / 6.0) - 1.0
        state["v150_projected_gain"] = projected
        if projected >= _V150_SCORE_GATE:
            state["v150_enabled"] = qualified
        print(
            "v150 load128 incumbent", inc_total, "candidate", cand_total,
            "ratio", load_ratio, "projected_score_gain", projected,
            "enabled", sorted(state["v150_enabled"]), flush=True,
        )
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v150 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["m"], cfg["layer"])
    if key in state.get("v150_enabled", ()):
        plan = state["v150"]["plans"].get(cfg["layer"])
        if plan is not None:
            return _v150_call(state["v150"], plan, data["x"])
    return _v105_kernel_fn(data, state)
'''

candidate = base + prelude + "\n\n" + pack_source + "\n\n" + integration.lstrip()
compile(candidate, str(OUT), "exec")
if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("upstream_sha256", UPSTREAM_SHA256)
    print("cuda_sha256", hashlib.sha256(cuda_source.encode()).hexdigest().upper())
    print("baseline_prefix", candidate.encode()[: len(base_bytes)] == base_bytes)
    print("bytes", len(candidate.encode()))
