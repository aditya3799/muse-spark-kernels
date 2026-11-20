"""Build S074 from protected S052: graft BF16 v150 RS128 QKV.

The generated candidate is self-contained.  It copies the proven Apache-2.0
RS128 producer/consumer kernel, exact C12 packer, escape tables, and validation
logic from the protected bf16_proj v150 component.  Only the decode layer's
T=128 fused-attention QKV branch is routed through it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
COMPONENT = ROOT.parent / "muse_bf16_proj" / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_s074_rs128_qkv.py"

BASE_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"
COMPONENT_SHA256 = "e405d6464ef5d64fdc896e300dd0f5ac9cd772d4e65f4eff7ff2338a2d0f7258"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("protected S052 hash mismatch; refusing to generate")
if _sha(COMPONENT) != COMPONENT_SHA256:
    raise RuntimeError("protected BF16 v150 hash mismatch; refusing to copy")

base = BASE.read_text(encoding="utf-8")
component = COMPONENT.read_text(encoding="utf-8")

# Copy the component implementation through its validation routine, excluding
# its bf16_proj-wide timing/routing wrapper.  The CUDA payload is the exact
# upstream source and retains the component's Apache-2.0 provenance header.
component_begin = component.index(
    "# ---------------------------------------------------------------- v150: faithful RS128 producer/consumer pipeline."
)
component_end = component.index("\ndef _v150_time(", component_begin)
rs128 = component[component_begin:component_end].rstrip()

# The copied section originally inherits this import from the bf16 component's
# preamble.  Make the generated decoder appendix self-contained.
rs128 = rs128.replace(
    "def _v150_build():\n",
    "def _v150_build():\n"
    "    from torch.utils.cpp_extension import load_inline as _s074_load_inline\n",
    1,
).replace(
    "_v150_module[0] = load_inline(",
    "_v150_module[0] = _s074_load_inline(",
    1,
)

qkv_anchor = '''        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
'''
qkv_replacement = '''        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if T == 128 and state.get("s074_rs128_enabled", False):
            qkv = _s074_rs128_run(state, data["attn_input"])
        elif qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
'''
if base.count(qkv_anchor) != 1:
    raise RuntimeError("protected fused-QKV dispatch anchor changed")
base = base.replace(qkv_anchor, qkv_replacement, 1)

fallback_anchor = '''    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    P, S = dense(x, state["qkv"], T)
'''
fallback_replacement = '''    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    if T == 128 and state.get("s074_rs128_enabled", False):
        P = _s074_rs128_run(state, x).view(1, T, state["qkv"].shape[0])
        S = 1
    else:
        P, S = dense(x, state["qkv"], T)
'''
if base.count(fallback_anchor) != 1:
    raise RuntimeError("protected fallback-QKV dispatch anchor changed")
base = base.replace(fallback_anchor, fallback_replacement, 1)

# S052 adds a later T=128/256 cuBLASLt QKV wrapper around the core function.
# Let the validated RS128 route fall through to the patched core at T=128;
# otherwise that wrapper preempts both QKV hooks above.
graft_anchor = '''    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
'''
graft_replacement = '''    if (plan is not None and not (T == 128 and state.get("s074_rs128_enabled", False))
            and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
'''
if base.count(graft_anchor) != 1:
    raise RuntimeError("protected cuBLASLt-QKV wrapper anchor changed")
base = base.replace(graft_anchor, graft_replacement, 1)

appendix = f'''

# ---------------------------------------------------------------------------
# S074: protected bf16_proj v150 RS128 pipeline, QKV-only at T=128.
# Component source SHA256: {COMPONENT_SHA256}
# ---------------------------------------------------------------------------
{rs128}


def _s074_rs128_run(state, x):
    vstate = state["s074_rs128"]
    return _v150_call(vstate, vstate["plan"], x)


_s074_prior_prepare = prepare
def prepare(weights):
    started = _v150_clock.time()
    dev = weights["qkv_gate"].device
    props = torch.cuda.get_device_properties(dev)
    eligible = props.major == 9 and "H200" in props.name
    build_thread = build_status = None
    if eligible:
        build_thread, build_status = _v150_start_build()

    state = _s074_prior_prepare(weights)
    state["s074_rs128_enabled"] = False
    if not eligible:
        return state

    try:
        # This matches the proven component's prepare-time ceiling and lets its
        # compile overlap the decoder's existing preparation work.
        build_thread.join(timeout=max(1.0, 220.0 - (_v150_clock.time() - started)))
        if not build_status or build_status[0] != "ok":
            print("s074 rs128 unavailable build", build_status or "not finished", flush=True)
            return state

        ext = _v150_build()
        weight = weights["qkv_gate"]
        packed = _v150_pack(weight)
        if packed is None:
            print("s074 rs128 rejected qkv pack unavailable", flush=True)
            return state
        lay, tab, esc6 = packed
        if not torch.equal(
            _v150_unpack(lay, tab, esc6, weight.shape[0], weight.shape[1]),
            weight.view(torch.int16).int() & 0xffff,
        ):
            print("s074 rs128 rejected qkv pack mismatch", flush=True)
            return state

        ks = 7
        esc, _ = _v150_esc(esc6, lay.shape[0], lay.shape[1], ks, dev)
        esc2, esc2_max = _v150_esc2(esc6, lay.shape[0], lay.shape[1], ks, dev)
        if esc2 is None:
            print("s074 rs128 rejected qkv esc2 overflow", esc2_max, flush=True)
            return state
        plan = {{
            "lay": lay, "tab": tab, "esc": esc, "esc2": esc2,
            "esc2_max": esc2_max,
            "cnt": torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev),
            "KS": ks, "n": weight.shape[0],
        }}
        vstate = {{
            "ext": ext,
            "prof": torch.empty(0, dtype=torch.int64, device=dev),
            "sink": torch.zeros(4, dtype=torch.int32, device=dev),
            "plan": plan,
        }}
        state["s074_rs128"] = vstate
        valid = _v150_validate(vstate, plan, weight, "qkv_gate")
        state["s074_rs128_enabled"] = bool(valid)
        print("s074 rs128 qkv enabled", bool(valid), flush=True)
    except Exception as exc:
        state["s074_rs128_enabled"] = False
        print("s074 rs128 unavailable", type(exc).__name__, str(exc), flush=True)
    return state
'''

candidate = base.rstrip() + "\n\n" + appendix.lstrip()
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
