"""Build S080: one exact-packed RS128 pooler+router projection at T=128.

S077 launches the 4096-row pooler RS128 kernel and the 256-row exact-FP32
router projection separately.  The decoder already owns a contiguous BF16
``pr`` matrix containing the pooler followed by the router's high and low
BF16 components, so this experiment gives all 4608 rows to one RS128 launch.

The projection output is BF16.  That is already the intended pooler type, but
rounding the two router dot products before their FP32 sum can theoretically
alter the top-8.  Therefore prepare enables this path only after:

* bit-exact reconstruction of all packed BF16 weights;
* the existing one-hot/counter/FP64 projection validation; and
* three deterministic 128-token probes preserving every ordered top-8 route.

Any failed guard leaves S077's split pooler plus exact-FP32 router untouched.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_s077_rs128_pdl.py"
OUT = ROOT / "submission_s080_rs128_pr_fused.py"

BASE_SHA256 = "437eada9f1ef0964b69fb804cadfe392e8adfbac72a98be65d75723a00d72178"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("validated S077 hash mismatch; refusing to extend")

base = BASE.read_text(encoding="utf-8")

old_route_guard = (
    "elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:"
)
new_route_guard = (
    "elif S == 1 and P.shape[-1] == 4608 and "
    "P.dtype in (torch.float32, torch.bfloat16):"
)
count = base.count(old_route_guard)
if count != 3:
    raise RuntimeError(f"combined route guards: expected 3 matches, found {count}")
base = base.replace(old_route_guard, new_route_guard)

append = r'''

# ---------------------------------------------------------------------------
# S080: one PDL RS128 projection for the T=128 pooler and router-high/router-low
# rows.  The BF16 combined route is enabled only after strict prepare-time
# projection and ordered top-8 agreement checks.
# ---------------------------------------------------------------------------


def _s080_route_probe(vstate, plan, weight, bias, seed):
    """Return ordered top-8 row agreement and router-logit relative error."""
    gen = torch.Generator(device=weight.device).manual_seed(seed)
    x = torch.randn((128, 8192), generator=gen, dtype=torch.float32,
                    device=weight.device).to(torch.bfloat16)
    got = _v150_call(vstate, plan, x)
    torch.cuda.synchronize(weight.device)

    approx = got[:, 4096:4352].float() + got[:, 4352:4608].float()
    ref = torch.mm(x, weight[4096:4352].T, out_dtype=torch.float32)
    ref.add_(torch.mm(x, weight[4352:4608].T, out_dtype=torch.float32))

    approx_sel = torch.sigmoid(approx * ROUTER_MULT) + bias
    ref_sel = torch.sigmoid(ref * ROUTER_MULT) + bias
    approx_ids = torch.topk(approx_sel, 8, dim=1, largest=True, sorted=True).indices
    ref_ids = torch.topk(ref_sel, 8, dim=1, largest=True, sorted=True).indices
    rows = float((approx_ids == ref_ids).all(dim=1).float().mean().item())
    rel = float(((approx - ref).norm() / ref.norm()).item())

    # A positive boundary margin is diagnostic evidence that the reference's
    # eighth choice is separated from the first rejected expert.
    top9 = torch.topk(ref_sel, 9, dim=1, largest=True, sorted=True).values
    min_gap = float((top9[:, 7] - top9[:, 8]).min().item())
    return rows, rel, min_gap


_s080_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    if (T == 128 and state.get("s080_pr_enabled", False)
            and tuple(h.shape) == (128, 8192) and h.dtype == torch.bfloat16
            and h.is_cuda and h.is_contiguous()):
        vstate = state["s074_rs128"]
        out = _v150_call(vstate, state["s080_pr_plan"], h, pdl=True)
        return out.view(1, T, 4608), 1
    return _s080_prior_pool_router_fp32(h, hr, T, state, ppick)


_s080_prior_prepare = prepare
def prepare(weights):
    state = _s080_prior_prepare(weights)
    state["s080_pr_enabled"] = False

    vstate = state.get("s074_rs128")
    if not state.get("s074_rs128_enabled", False) or vstate is None:
        print("s080 combined pr disabled: RS128 unavailable", flush=True)
        return state

    weight = state.get("pr")
    if (weight is None or tuple(weight.shape) != (4608, 8192)
            or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous()):
        print("s080 combined pr disabled: incompatible pr matrix", flush=True)
        return state

    try:
        packed = _v150_pack(weight)
        if packed is None:
            print("s080 combined pr rejected: pack unavailable", flush=True)
            return state
        lay, tab, esc6 = packed
        restored = _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1])
        exact = torch.equal(restored, weight.view(torch.int16).int() & 0xffff)
        if not exact:
            print("s080 combined pr rejected: pack mismatch", flush=True)
            return state

        ks = 4
        dev = weight.device
        esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
        esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
        if esc2 is None:
            print("s080 combined pr rejected: esc2 overflow", esc2_max, flush=True)
            return state
        plan = {
            "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
            "esc2_max": esc2_max,
            "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
            "KS": ks, "n": weight.shape[0],
        }
        if not _v150_validate(vstate, plan, weight, "combined_pr"):
            print("s080 combined pr rejected: projection validation", flush=True)
            return state

        probes = [
            _s080_route_probe(vstate, plan, weight, state["bias"], seed)
            for seed in (80128, 80129, 80130)
        ]
        route_exact = all(rows == 1.0 for rows, _, _ in probes)
        finite = all(
            torch.isfinite(torch.tensor((rows, rel, gap))).all().item()
            for rows, rel, gap in probes
        )
        # The relative bound is intentionally much tighter than the generic
        # projection check; exact route agreement remains the decisive guard.
        router_rel_ok = max(rel for _, rel, _ in probes) <= 6.0e-3
        enabled = bool(route_exact and finite and router_rel_ok)
        print("s080 combined pr exact_pack", exact, "route_probes", probes,
              "enabled", enabled, flush=True)
        if enabled:
            state["s080_pr_plan"] = plan
            state["s080_pr_enabled"] = True
    except Exception as exc:
        state["s080_pr_enabled"] = False
        state.pop("s080_pr_plan", None)
        print("s080 combined pr unavailable", type(exc).__name__, str(exc), flush=True)
    return state
'''

base = base.rstrip() + "\n" + append.lstrip("\n")
compile(base, str(OUT), "exec")
OUT.write_bytes(base.encode())
print(OUT)
print(hashlib.sha256(base.encode()).hexdigest())

