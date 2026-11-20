"""Build S076 from validated S075: add BF16 v150 RS128 pooler/unpooler.

The copied v150 translation unit already contains the exact producer/consumer
kernel used for S075's T=128 QKV route.  This appendix packs and validates the
two remaining proven matrices, then routes only the T=128 pooler and full
unpooler through them.  The router stays on the decoder's existing exact-FP32
BF16 hi+lo kernel; every other token count remains byte-for-byte S075.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_s075_cgemv_rs128.py"
OUT = ROOT / "submission_s076_rs128_pool_unpool.py"

BASE_SHA256 = "398dfa6195156898c51c29ac87464a77d8a7ff539f01b9f9b3b581be19b77b55"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("validated S075 hash mismatch; refusing to extend")

base = BASE.read_text(encoding="utf-8")

appendix = r'''

# ---------------------------------------------------------------------------
# S076: complete the protected bf16_proj v150 RS128 transfer at T=128.
# Pooler is paired with the existing exact-FP32 router; unpooler replaces the
# full 8192x4096 projection.  All plans reconstruct the BF16 weights exactly.
# ---------------------------------------------------------------------------


def _s076_rs128_run(state, layer, x):
    vstate = state["s074_rs128"]
    plan = vstate["plans"][layer]
    return _v150_call(vstate, plan, x)


_s076_prior_dense = dense
def dense(x, w, T, state=None, name=None):
    """Use the validated RS128 full unpooler only at the scored T=128 shape."""
    if (T == 128 and name == "un" and state is not None
            and "unpooler" in state.get("s076_rs128_enabled", ())
            and tuple(w.shape) == (8192, 4096) and tuple(x.shape) == (128, 4096)
            and x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()):
        out = _s076_rs128_run(state, "unpooler", x)
        return out.view(1, T, 8192), 1
    return _s076_prior_dense(x, w, T, state, name)


_s076_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    """RS128 BF16 pooler plus the decoder's exact-FP32 router at T=128."""
    if (T == 128 and "pooler" in state.get("s076_rs128_enabled", ())
            and state.get("bfp") is not None and tuple(h.shape) == (128, 8192)
            and h.dtype == torch.bfloat16 and h.is_cuda and h.is_contiguous()):
        pooled = _s076_rs128_run(state, "pooler", h)
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)
        return (pooled, logits), -1
    return _s076_prior_pool_router_fp32(h, hr, T, state, ppick)


_s076_prior_prepare = prepare
def prepare(weights):
    state = _s076_prior_prepare(weights)
    state["s076_rs128_enabled"] = set()

    # S075 has already compiled/validated the exact same CUDA translation unit
    # and QKV plan.  If that route is unavailable, preserve S075 unchanged.
    vstate = state.get("s074_rs128")
    if not state.get("s074_rs128_enabled", False) or vstate is None:
        print("s076 rs128 pool/unpool disabled: qkv plan unavailable", flush=True)
        return state

    plans = {"qkv_gate": vstate["plan"]}
    dev = weights["pooler"].device
    try:
        for layer in ("pooler", "unpooler"):
            weight = weights[layer]
            packed = _v150_pack(weight)
            if packed is None:
                print("s076 rs128 rejected", layer, "pack unavailable", flush=True)
                continue
            lay, tab, esc6 = packed
            restored = _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1])
            exact = torch.equal(restored, weight.view(torch.int16).int() & 0xffff)
            if not exact:
                print("s076 rs128 rejected", layer, "pack mismatch", flush=True)
                continue

            ks = _V150_KS[layer]
            esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
            esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
            if esc2 is None:
                print("s076 rs128 rejected", layer, "esc2 overflow", esc2_max, flush=True)
                continue
            plan = {
                "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
                "esc2_max": esc2_max,
                "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
                "KS": ks, "n": weight.shape[0],
            }
            plans[layer] = plan
            if _v150_validate(vstate, plan, weight, layer):
                state["s076_rs128_enabled"].add(layer)
            else:
                del plans[layer]

        vstate["plans"] = plans
        print("s076 rs128 enabled", sorted(state["s076_rs128_enabled"]), flush=True)
    except Exception as exc:
        state["s076_rs128_enabled"] = set()
        vstate["plans"] = {"qkv_gate": vstate["plan"]}
        print("s076 rs128 pool/unpool unavailable", type(exc).__name__, str(exc), flush=True)
    return state
'''

candidate = base.rstrip() + "\n\n" + appendix.lstrip()
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
