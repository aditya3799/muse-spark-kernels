"""Generate v155: pooler RS128 two-resident-CTA output decomposition.

Case 0 is byte-for-byte v150.  Case 1 changes only the CTA/output geometry:
each CTA owns 64 output rows (HALF) and uses the 256-thread H2 launch bound,
while retaining NT=128, ST=5, KS=4, C12, CURP, WGMMA, and the incumbent
global BF16 reduce-add epilogue.  The doubled x-grid gives 256 CTAs, intended
to place two independent CTAs on each H200 SM and hide consumer-side stalls.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_half_h2_v155.py"

# v150 d2s5hzwP flags plus HALF (bit 4) and H2 (bit 10).
V155_FLAGS = 42999939 | 16 | 1024
half_case = (
    v150.case
    + f"\n    case 1: return rq_go<128, 5, 1, 0, 0, 0, {V155_FLAGS}, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
if v150.cuda_source.count(v150.case) != 1:
    raise RuntimeError("v150 case anchor changed")
half_cuda = v150.cuda_source.replace(v150.case, half_case)
half_payload = base64.b64encode(zlib.compress(half_cuda.encode(), 9)).decode()

append = f'''

# ---------------------------------------------------------------- v155: 64-row, two-resident-CTA pooler m=128.
_V150_CUDA_PAYLOAD = "{half_payload}"
_V155_CUDA_SHA256 = "{hashlib.sha256(half_cuda.encode()).hexdigest().upper()}"
_V155_FLAGS = {V155_FLAGS}
_V155_ROUTE_RATIO = 0.97


def _v155_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v155_rs128_half_h2_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
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


_v150_build = _v155_build
_v155_v150_prepare = prepare
_v155_v150_kernel_fn = kernel_fn


def _v155_call(state, plan, variant, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        variant, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v155 variant %d rqrun error %d" % (variant, rc))
    return out


def _v155_info(state, plan, variant, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        variant, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v155_validate(state, plan, weight):
    variant, m, k, ks = 1, 128, weight.shape[1], plan["KS"]
    cols = {{0, k - 1, k // 3, (2 * k) // 3}}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    plan["cnt"].zero_()
    out = _v155_call(state, plan, variant, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    clean = int(plan["cnt"].abs().sum()) == 0
    if not exact or not clean:
        plan["cnt"].zero_()
        print("v155 rejected HALF-H2 one-hot/counter", exact, clean, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(155128)
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v155_call(state, plan, variant, x)
    torch.cuda.synchronize()
    ref = x[:8].double() @ weight.double().T
    diff = out[:8].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    clean = int(plan["cnt"].abs().sum()) == 0
    regs = _v155_info(state, plan, variant, 1)
    local = _v155_info(state, plan, variant, 2)
    smem = _v155_info(state, plan, variant, 3)
    occupancy = _v155_info(state, plan, variant, 4)
    ok = (
        bool(torch.isfinite(out).all()) and rel <= 6e-3 and worst <= 3e-2
        and clean and local == 0 and occupancy >= 2
    )
    if not clean:
        plan["cnt"].zero_()
    print(
        "v155 exact pooler HALF-H2 one_hot", exact, "fp64", rel, worst,
        "regs", regs, "local", local, "smem", smem,
        "ctas_per_sm", occupancy, "valid", ok, flush=True,
    )
    return ok


def prepare(weights):
    state = _v155_v150_prepare(weights)
    state["v155_half_h2_enabled"] = False
    try:
        dev = weights["pooler"].device
        props = torch.cuda.get_device_properties(dev)
        vstate = state.get("v150")
        plan = None if vstate is None else vstate.get("plans", {{}}).get("pooler")
        if props.major != 9 or "H200" not in props.name or plan is None:
            return state

        weight = weights["pooler"]
        if not _v155_validate(vstate, plan, weight):
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(155130)
        x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
        base_out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16, device=dev)
        cand_out = torch.empty_like(base_out)
        ratio, cand_us, base_us = _v150_pair(
            lambda: _v155_call(vstate, plan, 1, x, cand_out),
            lambda: _v150_call(vstate, plan, x, base_out),
            flush,
        )
        keep = cand_us < base_us and ratio <= _V155_ROUTE_RATIO
        print(
            "v155 pooler HALF-H2 candidate", cand_us,
            "v150_full", base_us, "core_ratio", ratio,
            "qualified", keep, flush=True,
        )
        if keep:
            state["v155_half_h2_enabled"] = True
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v155 HALF-H2 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (state.get("v155_half_h2_enabled") and cfg["m"] == 128 and
            cfg["layer"] == "pooler"):
        vstate = state["v150"]
        plan = vstate["plans"].get("pooler")
        if plan is not None:
            return _v155_call(vstate, plan, 1, data["x"])
    return _v155_v150_kernel_fn(data, state)
'''

base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("half_h2_cuda_sha256", hashlib.sha256(half_cuda.encode()).hexdigest().upper())
print("flags", V155_FLAGS)
print("bytes", len(candidate.encode()))
