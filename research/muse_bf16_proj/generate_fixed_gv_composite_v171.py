from pathlib import Path
import base64
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_composite_v170.py"
GV82 = ROOT / "submission_gv_noescape_probe_v82.py"
OUTPUT = ROOT / "submission_fixed_gv_composite_v171.py"


text = BASE.read_text()
payload = base64.b64encode(zlib.compress(GV82.read_bytes(), 9)).decode()

append = r'''

# ---------------------------------------------------------------- v171: fixed small-shape C12 composition.
# v82's H200 run supplied repeatable wins on the small load buckets, but its
# wide routes predate the protected v150 kernels.  Load the implementation in
# an isolated namespace and compile only the nine winning GVI instances plus
# the independently validated O-32 direct kernel.  No exhaustive tuner or old
# fallback route is reachable from dispatch.
import base64 as _v171_b64
import zlib as _v171_zlib
import types as _v171_types
import threading as _v171_threading

_V171_GV82_PAYLOAD = "@@PAYLOAD@@"
_v171_gv = _v171_types.ModuleType("_v171_fixed_gv")
_v171_gv.__file__ = __file__
exec(compile(_v171_zlib.decompress(_v171_b64.b64decode(_V171_GV82_PAYLOAD)).decode(),
             "_v171_fixed_gv", "exec"), _v171_gv.__dict__)

_V171_INSTANCES = {
    ("qkv_gate", 1):  (1, 1, 1, 1, 4, 8, 4, 1, 2, 5),
    ("qkv_gate", 8):  (1, 1, 1, 1, 4, 8, 4, 1, 2, 13),
    ("qkv_gate", 16): (1, 2, 1, 1, 4, 8, 4, 1, 2, 5),
    ("o", 1):         (1, 1, 2, 1, 8, 2, 1, 1, 0, 1),
    ("o", 16):        (1, 2, 2, 1, 8, 2, 1, 1, 0, 0),
    ("pooler", 16):   (1, 2, 2, 2, 8, 16, 1, 1, 0, 1),
    ("router", 8):    (2, 1, 2, 1, 8, 2, 8, 1, 0, 4),
    ("router", 16):   (2, 2, 2, 1, 8, 2, 8, 1, 0, 4),
    ("router", 32):   (2, 4, 2, 1, 4, 2, 16, 1, 0, 0),
}
_V171_ROUTES = frozenset(_V171_INSTANCES) | frozenset({("o", 32)})


def _v171_material(gv, weights, layer):
    w = weights[layer]
    if layer == "router":
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        frs = [gv._pack(hi), gv._pack(lo)]
        return {
            "weight": w, "frs": frs, "exact": hi.float() + lo.float(),
            "css": None, "esc": None,
        }
    frs = [gv._pack(w)]
    css = [[gv._c12(f, dv) for f in frs] for dv in (0, 1)]
    if not all(c is not None for cs in css for c in cs):
        raise RuntimeError("C12 pack unavailable for " + layer)
    esc = torch.cat([
        torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1)
        for h, c in enumerate(css[1])
    ])
    return {"weight": w, "frs": frs, "exact": w, "css": css, "esc": esc}


def _v171_make_plan(gv, state, material, layer, bucket, inst):
    frs = material["frs"]
    tf = frs[0].shape[0]
    n = tf * 16
    geo = inst[3:9]
    if layer == "router":
        src = frs
        tables = (0, 0)
        esc = state["noesc"]
    else:
        cs = material["css"][inst[-1] & 1]
        src = [c[0] for c in cs]
        tables = (cs[0][1], cs[-1][1])
        esc = gv._gvesc(material["esc"], tf, geo)
        if esc is None:
            raise RuntimeError("escape list overflow for %s m%d" % (layer, bucket))
    index = gv.GVI.index(inst)
    plan = (index, src[0], esc, tables[0], n, src[-1], tables[1])
    if not gv._gvok(state, plan, material["exact"], material["weight"], bucket):
        raise RuntimeError("fixed GVI validation failed for %s m%d" % (layer, bucket))
    return plan


def _v171_prepare_fixed(weights):
    gv = _v171_gv
    dev = weights["qkv_gate"].device
    state = {
        "plans": {},
        "gvws": torch.zeros((1 << 24,), dtype=torch.float32, device=dev),
        "gvcnt": torch.zeros((1 << 16,), dtype=torch.int32, device=dev),
        "wrcnt": torch.zeros((4096,), dtype=torch.int32, device=dev),
        "noesc": torch.full((1 << 16,), -1, dtype=torch.int32, device=dev),
    }

    state["o32"] = gv._o32_prepare(weights)

    materials = {
        layer: _v171_material(gv, weights, layer)
        for layer in sorted({key[0] for key in _V171_INSTANCES})
    }
    for (layer, bucket), inst in _V171_INSTANCES.items():
        state["plans"][(layer, bucket)] = _v171_make_plan(
            gv, state, materials[layer], layer, bucket, inst)
    print("v171 fixed routes validated", sorted(state["plans"]),
          "o32", bool(state["o32"].get("enabled")), flush=True)
    return state


_v171_base_prepare = prepare
_v171_base_kernel_fn = kernel_fn


def prepare(weights):
    # Compile both added extensions while the protected lineage performs its
    # own packing, validation, and background build.  This keeps total prepare
    # wall time bounded without changing any GPU-side selection decision.
    _v171_gv.GVI = list(dict.fromkeys(_V171_INSTANCES.values()))
    _v171_gv._gvmod = None
    build_status = []
    def build_one(label, fn):
        try:
            fn()
            build_status.append((label, "ok"))
        except Exception as exc:
            build_status.append((label, "failed: " + str(exc).splitlines()[-1][:240]))
    builds = [
        _v171_threading.Thread(target=build_one, args=("gv", _v171_gv._gvext), daemon=True),
        _v171_threading.Thread(target=build_one, args=("o32", _v171_gv._o32_ext), daemon=True),
    ]
    for thread in builds:
        thread.start()
    state = _v171_base_prepare(weights)
    for thread in builds:
        thread.join()
    state["v171"] = None
    try:
        if sorted(build_status) != [("gv", "ok"), ("o32", "ok")]:
            raise RuntimeError("extension build " + repr(build_status))
        state["v171"] = _v171_prepare_fixed(weights)
    except Exception as exc:
        print("v171 unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    vstate = state.get("v171")
    if vstate is not None and (layer, m) in _V171_ROUTES:
        if layer == "o" and m == 32:
            out = _v171_gv._o32_run(data["x"].contiguous(), layer, n, vstate["o32"])
            if out is not None:
                return out
        plan = vstate["plans"].get((layer, m))
        if plan is not None:
            out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16,
                              device=data["x"].device)
            rc = _v171_gv._gvcall(vstate, plan, data["x"].contiguous(), out)
            if rc:
                raise RuntimeError("v171 GVI launch failed: %d" % rc)
            return out
    return _v171_base_kernel_fn(data, state)
'''

text += append.replace("@@PAYLOAD@@", payload)
OUTPUT.write_text(text)
print(OUTPUT)
