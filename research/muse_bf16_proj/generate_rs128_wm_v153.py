"""Generate v153: pooler RS128 WGMMA/decode overlap schedule bake.

Case 0 is byte-for-byte v150. Cases 1 and 2 change only WM: decode the next
C12 stage while the current WGMMA group is in flight, then use wait_group 0
(WM1) or keep one group in flight with triple-buffered A registers (WM2).
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_wm_v153.py"

wm_case = (
    v150.case
    + "\n    case 1: return rq_go<128, 5, 1, 0, 1, 0, 42999939, 1>"
      "(p, x, out, (int)KS, (int)info);"
    + "\n    case 2: return rq_go<128, 5, 1, 0, 2, 0, 42999939, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
if v150.cuda_source.count(v150.case) != 1:
    raise RuntimeError("v150 case anchor changed")
wm_cuda = v150.cuda_source.replace(v150.case, wm_case)
wm_payload = base64.b64encode(zlib.compress(wm_cuda.encode(), 9)).decode()

append = f'''

# ---------------------------------------------------------------- v153: overlap C12 decode with WGMMA for pooler m=128.
_V150_CUDA_PAYLOAD = "{wm_payload}"
_V153_CUDA_SHA256 = "{hashlib.sha256(wm_cuda.encode()).hexdigest().upper()}"
_V153_ROUTE_RATIO = 0.97


def _v153_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v153_rs128_wm_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
        _v150_module[0] = load_inline(
            tag,
            cpp_sources=_V150_CPP,
            cuda_sources=src,
            functions=["rqrun"],
            extra_cuda_cflags=[
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-lineinfo",
            ],
            verbose=False,
        )
    return _v150_module[0]


_v150_build = _v153_build
_v153_v150_prepare = prepare
_v153_v150_kernel_fn = kernel_fn


def _v153_call(state, plan, variant, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        variant, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v153 WM%d rqrun error %d" % (variant, rc))
    return out


def _v153_info(state, plan, variant, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        variant, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v153_validate(state, plan, weight, variant):
    m, k, ks = 128, weight.shape[1], plan["KS"]
    cols = {{0, k - 1, k // 3, (2 * k) // 3}}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    plan["cnt"].zero_()
    out = _v153_call(state, plan, variant, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    clean = int(plan["cnt"].abs().sum()) == 0
    if not exact or not clean:
        plan["cnt"].zero_()
        print("v153 rejected WM", variant, "one-hot/counter", exact, clean, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(153128 + variant)
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v153_call(state, plan, variant, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    clean = int(plan["cnt"].abs().sum()) == 0
    ok = bool(torch.isfinite(out).all()) and rel <= 6e-3 and worst <= 3e-2 and clean
    if not clean:
        plan["cnt"].zero_()
    print(
        "v153 exact pooler WM", variant, "one_hot", exact, "fp64", rel, worst,
        "regs", _v153_info(state, plan, variant, 1),
        "local", _v153_info(state, plan, variant, 2),
        "smem", _v153_info(state, plan, variant, 3), "valid", ok, flush=True,
    )
    return ok


def prepare(weights):
    state = _v153_v150_prepare(weights)
    state["v153_wm_enabled"] = False
    try:
        dev = weights["pooler"].device
        props = torch.cuda.get_device_properties(dev)
        vstate = state.get("v150")
        plan = None if vstate is None else vstate.get("plans", {{}}).get("pooler")
        if props.major != 9 or "H200" not in props.name or plan is None:
            return state

        weight = weights["pooler"]
        valid = [v for v in (1, 2) if _v153_validate(vstate, plan, weight, v)]
        if not valid:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(153130)
        x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
        base_out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16, device=dev)
        results = []
        for variant in valid:
            cand_out = torch.empty_like(base_out)
            ratio, cand_us, base_us = _v150_pair(
                lambda variant=variant, cand_out=cand_out: _v153_call(vstate, plan, variant, x, cand_out),
                lambda: _v150_call(vstate, plan, x, base_out),
                flush,
            )
            keep = cand_us < base_us and ratio <= _V153_ROUTE_RATIO
            results.append((ratio, cand_us, base_us, variant, keep))
            print(
                "v153 pooler WM", variant, "candidate", cand_us,
                "v150_WM0", base_us, "core_ratio", ratio,
                "qualified", keep, flush=True,
            )
        winners = [item for item in results if item[4]]
        if winners:
            best = min(winners, key=lambda item: (item[0], item[1]))
            vstate["wm_pooler_variant"] = best[3]
            state["v153_wm_enabled"] = True
            print("v153 pooler picked WM", best[3], flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v153 WM unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (state.get("v153_wm_enabled") and cfg["m"] == 128 and
            cfg["layer"] == "pooler"):
        vstate = state["v150"]
        plan = vstate["plans"].get("pooler")
        variant = vstate.get("wm_pooler_variant")
        if plan is not None and variant is not None:
            return _v153_call(vstate, plan, variant, data["x"])
    return _v153_v150_kernel_fn(data, state)
'''

base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("wm_cuda_sha256", hashlib.sha256(wm_cuda.encode()).hexdigest().upper())
print("bytes", len(candidate.encode()))
