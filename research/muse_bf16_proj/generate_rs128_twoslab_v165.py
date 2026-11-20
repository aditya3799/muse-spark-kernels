"""Generate v165: exact RS128 m=256 as two one-wave token slabs.

Each CTA keeps v150's proven 384-thread/168-register NT128 schedule.  A z-grid
dimension selects one of two 128-token slabs, while route-specific K-splits
keep the complete grid at or below 128 resident CTAs.  This trades duplicated
compressed-weight requests for the strong v150 consumer schedule and lets L2
coalesce/reuse the identical weight slabs across token peers.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import zlib

import generate_rs128_faithful_v150 as v150


OUT = v150.ROOT / "submission_rs128_twoslab_v165.py"
TOK2 = 1 << 27
flags = v150.variant[7] | TOK2

cuda = v150.cuda_source


def replace_once(old: str, new: str) -> None:
    global cuda
    count = cuda.count(old)
    if count != 1:
        raise RuntimeError(f"expected one CUDA match, found {count}: {old[:100]!r}")
    cuda = cuda.replace(old, new, 1)


replace_once(
    "  constexpr bool HALF = (FL & 16) != 0;\n",
    "  constexpr bool HALF = (FL & 16) != 0;\n"
    "  constexpr bool TOK2 = (FL & 134217728) != 0;  // v165: two independent NT128 token slabs\n",
)
replace_once(
    "  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;\n",
    "  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;\n"
    "  const int mt0 = TOK2 ? (int)blockIdx.z * NT : 0;\n"
    "  const int cidx = (TOK2 ? (int)blockIdx.z * (int)gridDim.x : 0) + (int)blockIdx.x;\n"
    "  const int ncidx = (int)gridDim.x * (TOK2 ? (int)gridDim.z : 1);\n",
)

# Every x TMA uses the selected token slab as its row coordinate.
if cuda.count(", 0, full + 8 *") != 6:
    raise RuntimeError("unexpected x-TMA row-coordinate count")
cuda = cuda.replace(", 0, full + 8 *", ", mt0, full + 8 *")

replace_once(
    "  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;\n",
    "  const unsigned long long cta = ((unsigned long long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;\n",
)
replace_once(
    "      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);\n",
    "      const int t = tid - (PW + 2) * 32, r0 = mt0 + (int)((long long)ks * NT / KS), r1 = mt0 + (int)((long long)(ks + 1) * NT / KS);\n",
)
cuda = cuda.replace("p.cnt + blockIdx.x", "p.cnt + cidx")
replace_once(
    "if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }",
    "if (atomicAdd(p.cnt + ncidx + cidx, 1) == KS - 1) { p.cnt[cidx] = 0; p.cnt[ncidx + cidx] = 0; }",
)

# Output TMA coordinates are likewise relative to the selected token slab.
if cuda.count('"r"(tok0), "r"(src)') != 2:
    raise RuntimeError("unexpected EH output-coordinate count")
cuda = cuda.replace('"r"(tok0), "r"(src)', '"r"(mt0 + tok0), "r"(src)')
if cuda.count('"r"(0), "r"(slab)') != 2:
    raise RuntimeError("unexpected regular output-coordinate count")
cuda = cuda.replace('"r"(0), "r"(slab)', '"r"(mt0), "r"(slab)')

replace_once(
    "  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;\n",
    "  constexpr bool TOK2 = (FL & 134217728) != 0;\n"
    "  if (p.m < 1 || (TOK2 ? p.m != 2 * NT : p.m > NT) || KS < 1 || p.KC / KS < GS || KS > 64) return -4;\n",
)
replace_once(
    "  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;\n",
    "  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1) * (TOK2 ? 2 : 1)) return -5;\n",
)
replace_once(
    "  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);\n",
    "  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS, TOK2 ? 2 : 1), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);\n",
)
replace_once(
    "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);\n",
    "    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);\n"
    f"    case 1: return rq_go<128, 5, 1, 0, 0, 0, {flags}, 1>(p, x, out, (int)KS, (int)info);\n",
)

payload = base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()
prelude = v150.prelude.replace(v150.cuda_payload, payload)

integration = r'''

# ---------------------------------------------------------------- v165: two NT128 slabs at m=256.
_V165_TARGETS = ("qkv_gate", "o", "pooler", "unpooler")
_V165_KS = {"qkv_gate": 3, "o": 1, "pooler": 2, "unpooler": 1}
_V165_ROUTE_RATIO = 0.97
_V165_SCORE_GATE = 0.009


def _v165_call(vstate, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = vstate["ext"].rqrun(
        1, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v165 rqrun error %d" % rc)
    return out


def _v165_info(vstate, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(vstate["ext"].rqrun(
        1, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, vstate["prof"], vstate["sink"], kind,
    ))


def _v165_validate(vstate, plan, weight, layer):
    m, k, ks = 256, weight.shape[1], plan["KS"]
    cols = {0, k - 1, k // 3, (2 * k) // 3}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
        x[128 + row, col] = 8.0
    out = _v165_call(vstate, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and torch.equal(out[128:128 + len(cols)], want)
    exact = exact and bool((out[len(cols)] == 0).all()) and bool((out[128 + len(cols)] == 0).all())
    if not exact or int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_()
        print("v165 rejected", layer, "one-hot/counter", exact, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(1650 + len(layer))
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v165_call(vstate, plan, x)
    torch.cuda.synchronize()
    rows = torch.tensor([0, 1, 2, 3, 128, 129, 130, 131], device=weight.device)
    ref = x[rows].double() @ weight.double().T
    diff = out[rows].double() - ref
    rel = float(diff.norm() / ref.norm())
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    tol = _V150_TOL.get(ks, 8e-3)
    regs, local, smem = (_v165_info(vstate, plan, i) for i in (1, 2, 3))
    ok = bool(torch.isfinite(out).all()) and rel <= tol and worst <= 5 * tol and local == 0
    if int(plan["cnt"].abs().sum()):
        plan["cnt"].zero_(); ok = False
    print("v165 exact", layer, "one_hot", exact, "fp64", rel, worst,
          "regs", regs, "local", local, "smem", smem,
          "esc2_max", plan["esc2_max"], "valid", ok, flush=True)
    return ok


_v165_base_prepare = prepare
_v165_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v165_base_prepare(weights)
    state["v165_enabled"] = set()
    state["v165_projected_gain"] = 0.0
    vstate = state.get("v150")
    if vstate is None:
        return state
    dev = weights["pooler"].device
    try:
        plans = {}
        for layer in _V165_TARGETS:
            weight = weights[layer]
            packed = _v150_pack(weight)
            if packed is None:
                continue
            lay, tab, esc6 = packed
            if not torch.equal(_v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
                               weight.view(torch.int16).int() & 0xffff):
                continue
            ks = _V165_KS[layer]
            esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
            esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
            if esc2 is None:
                continue
            plan = {
                "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                "esc2_max": esc2_max,
                "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                "KS": ks, "n": weight.shape[0],
            }
            if _v165_validate(vstate, plan, weight, layer):
                plans[layer] = plan
        state["v165_plans"] = plans
        if not plans:
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(165256)
        inc_total = cand_total = 0.0
        qualified = set()
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            weight = weights[layer]
            x = torch.randn((256, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            cfg = {"layer": layer, "m": 256, "n": weight.shape[0], "k": weight.shape[1]}
            plan = plans.get(layer)
            if plan is None:
                unchanged = _v150_time(lambda x=x, cfg=cfg: _v165_base_kernel_fn({"x": x, "config": cfg}, state), flush)
                core = max(unchanged - _V150_REPLAY_US, 0.05)
                inc_total += core; cand_total += core
                continue
            out = torch.empty((256, weight.shape[0]), dtype=torch.bfloat16, device=dev)
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda x=x, out=out, plan=plan: _v165_call(vstate, plan, x, out),
                lambda x=x, cfg=cfg: _v165_base_kernel_fn({"x": x, "config": cfg}, state),
                flush,
            )
            keep = candidate_us < incumbent_us and ratio <= _V165_ROUTE_RATIO
            if keep:
                qualified.add(layer)
            inc_core = max(incumbent_us - _V150_REPLAY_US, 0.05)
            cand_core = max(candidate_us - _V150_REPLAY_US, 0.05) if keep else inc_core
            inc_total += inc_core; cand_total += cand_core
            print("v165 rs256", layer, "candidate", candidate_us, "incumbent", incumbent_us,
                  "core_ratio", ratio, "qualified", keep, flush=True)

        projected = (inc_total / cand_total) ** (1.0 / 6.0) - 1.0
        state["v165_projected_gain"] = projected
        if projected >= _V165_SCORE_GATE:
            state["v165_enabled"] = qualified
        print("v165 load256 incumbent", inc_total, "candidate", cand_total,
              "projected_score_gain", projected,
              "enabled", sorted(state["v165_enabled"]), flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v165 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if cfg["m"] == 256 and cfg["layer"] in state.get("v165_enabled", ()):
        plan = state.get("v165_plans", {}).get(cfg["layer"])
        if plan is not None:
            return _v165_call(state["v150"], plan, data["x"])
    return _v165_base_kernel_fn(data, state)
'''

candidate = (
    v150.base + prelude + "\n\n" + v150.pack_source + "\n\n"
    + v150.integration.lstrip() + "\n\n" + integration.lstrip()
)
compile(candidate, str(OUT), "exec")

if __name__ == "__main__":
    OUT.write_bytes(candidate.encode())
    print(OUT)
    print("variant", (128, 5, 1, 0, 0, 0, flags, 1))
    print("cuda_sha256", hashlib.sha256(cuda.encode()).hexdigest().upper())
    print("candidate_sha256", hashlib.sha256(candidate.encode()).hexdigest().upper())
