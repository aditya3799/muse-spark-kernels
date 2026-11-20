"""Generate v152: raw-BF16 register-A twin of v150's pooler RS128 path.

The scored v150 prefix remains byte-identical.  Case 0 is the unchanged C12
kernel.  Case 1 keeps the same RS128 topology but streams the already arranged
BF16 A fragments, removing C12 decode and escape handling at the cost of 4/3
as many weight bytes.  Only pooler m=128 is baked against case 0.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_raw_v152.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return source.replace(old, new)


raw_cuda = v150.cuda_source
raw_cuda = replace_once(
    raw_cuda,
    "  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA",
    "  static constexpr int XB = NT * 128, WB = (FL & (1 << 26)) ? 16384 : ((FL & 16) ? 6144 : 12288), ND = NT / 2;  // bit 26: raw BF16 A fragments",
    "raw WB",
)
raw_cuda = replace_once(
    raw_cuda,
    "  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;",
    "  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;\n  constexpr bool RAW = (FL & (1 << 26)) != 0;  // pre-arranged BF16 WGMMA A registers",
    "RAW flag",
)

# All producer addresses use the compile-time bytes-per-stage.  These are five
# executable occurrences in the upstream source (three base pointers and two
# copy addresses in each of the early/split/steady producer paths).
if raw_cuda.count("* 12288") != 9:
    raise RuntimeError(f"W stride: expected nine occurrences, found {raw_cuda.count('* 12288')}")
raw_cuda = raw_cuda.replace("* 12288", "* WB")
if raw_cuda.count("hh * 6144") != 3:
    raise RuntimeError(f"half offset: expected three occurrences, found {raw_cuda.count('hh * 6144')}")
raw_cuda = raw_cuda.replace("hh * 6144", "hh * (WB / 2)")

raw_cuda = replace_once(
    raw_cuda,
    "  const uint32_t wl = wring + warp * 1536 + lane * 16;",
    "  const uint32_t wl = wring + warp * (WB / NW) + lane * 16;",
    "per-warp W slice",
)
raw_cuda = replace_once(
    raw_cuda,
    "  uint4 Q[GS][3];\n#pragma unroll\n  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);",
    "  uint4 Q[GS][3], R[GS];\n#pragma unroll\n  for (int j = 0; j < GS; j++) { Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0); R[j] = make_uint4(0, 0, 0, 0); }",
    "raw LDS registers",
)
raw_cuda = replace_once(
    raw_cuda,
    "          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);",
    "          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);\n          if constexpr (RAW) R[j] = rq_lds(a + 1536);",
    "fourth raw LDS",
)
raw_cuda = replace_once(
    raw_cuda,
    "        if (PRB == 10) {\n#pragma unroll\n          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;\n        } else {",
    "        if constexpr (RAW) {\n          A[j][0] = Q[j][0].x; A[j][1] = Q[j][0].y; A[j][2] = Q[j][0].z; A[j][3] = Q[j][0].w;\n          A[j][4] = Q[j][1].x; A[j][5] = Q[j][1].y; A[j][6] = Q[j][1].z; A[j][7] = Q[j][1].w;\n          A[j][8] = Q[j][2].x; A[j][9] = Q[j][2].y; A[j][10] = Q[j][2].z; A[j][11] = Q[j][2].w;\n          A[j][12] = R[j].x; A[j][13] = R[j].y; A[j][14] = R[j].z; A[j][15] = R[j].w;\n        } else if (PRB == 10) {\n#pragma unroll\n          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;\n        } else {",
    "raw register copy",
)
raw_cuda = replace_once(
    raw_cuda,
    "    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;",
    "    if (W.dim() != 4 || W.size(2) != 8 || (W.size(3) != 1536 && W.size(3) != 2048) || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;",
    "raw host layout",
)

RAW_FLAGS = (42999939 & ~((1 << 13) | (1 << 20) | (1 << 23) | (1 << 25))) | (1 << 26)
if RAW_FLAGS != 67108995:
    raise RuntimeError(f"unexpected raw flags {RAW_FLAGS}")
raw_case = (
    v150.case
    + f"\n    case 1: return rq_go<128, 5, 1, 0, 0, 0, {RAW_FLAGS}, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
raw_cuda = replace_once(raw_cuda, v150.case, raw_case, "raw dispatch case")
raw_payload = base64.b64encode(zlib.compress(raw_cuda.encode(), 9)).decode()


append = f'''

# ---------------------------------------------------------------- v152: raw-BF16 pooler m=128 RS128 twin.
# Case 0 is v150's unchanged lossless C12 path.  Case 1 uses the same pipeline
# with 16 KB raw BF16 A fragments per 128-row k64 stage and no decode/escapes.
_V150_CUDA_PAYLOAD = "{raw_payload}"
_V152_CUDA_SHA256 = "{hashlib.sha256(raw_cuda.encode()).hexdigest().upper()}"
_V152_RAW_FLAGS = {RAW_FLAGS}
_V152_ROUTE_RATIO = 0.97


def _v152_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v152_rs128_raw_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
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


_v150_build = _v152_build
_v152_v150_prepare = prepare
_v152_v150_kernel_fn = kernel_fn


def _v152_raw_pack(weight):
    n, k = weight.shape
    if n % 128 or k % 64:
        return None
    natural = _v150_pack_nat(weight)
    # A k64 fragment for one 16-row tile is 4 * 32 * 8 BF16 = 2048 B.
    blocked = natural.reshape(n // 128, 8, k // 64, 1024)
    blocked = blocked.permute(0, 2, 1, 3).contiguous()
    return blocked.view(torch.uint8)


def _v152_raw_unpack(lay, n, k):
    fragments = lay.view(torch.bfloat16).permute(0, 2, 1, 3)
    fragments = fragments.reshape(n // 16, k // 64, 4, 32, 8)
    w = fragments.view(n // 16, k // 64, 4, 8, 4, 2, 2, 2)
    return w.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(n, k)


def _v152_raw_call(state, plan, x, out=None, info=0):
    if out is None:
        out = torch.empty((x.shape[0], plan["n"]), dtype=torch.bfloat16, device=x.device)
    rc = state["ext"].rqrun(
        1, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], info,
    )
    if rc != 0:
        raise RuntimeError("v152 raw rqrun error %d" % rc)
    return out


def _v152_raw_info(state, plan, kind):
    empty_bf = torch.empty(0, dtype=torch.bfloat16, device=plan["lay"].device)
    empty_i = torch.empty(0, dtype=torch.int32, device=plan["lay"].device)
    return int(state["ext"].rqrun(
        1, empty_bf, empty_bf, empty_bf, empty_i, empty_i, empty_i,
        plan["KS"], plan["tab"], 0, state["prof"], state["sink"], kind,
    ))


def _v152_raw_validate(state, plan, weight):
    m, n, k, ks = 128, weight.shape[0], weight.shape[1], plan["KS"]
    cols = {{0, k - 1, k // 3, (2 * k) // 3}}
    for split in range(1, ks):
        boundary = k * split // ks
        cols.update((max(0, boundary - 1), min(k - 1, boundary)))
    cols = sorted(cols)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
    for row, col in enumerate(cols):
        x[row, col] = 8.0
    plan["cnt"].zero_()
    out = _v152_raw_call(state, plan, x)
    torch.cuda.synchronize()
    want = torch.stack([(weight[:, col] * 8.0).to(torch.bfloat16) for col in cols])
    exact = torch.equal(out[:len(cols)], want) and bool((out[len(cols)] == 0).all())
    clean = int(plan["cnt"].abs().sum()) == 0
    if not exact or not clean:
        plan["cnt"].zero_()
        print("v152 raw rejected pooler one-hot/counter", exact, clean, flush=True)
        return False

    gen = torch.Generator(device=weight.device).manual_seed(152128)
    x = torch.randn((m, k), generator=gen, dtype=torch.float32, device=weight.device)
    hot = torch.randperm(k, generator=gen, device=weight.device)[:max(1, k // 100)]
    x[:, hot] *= 30.0
    x = x.to(torch.bfloat16)
    out = _v152_raw_call(state, plan, x)
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
        "v152 raw exact pooler", "one_hot", exact, "fp64", rel, worst,
        "regs", _v152_raw_info(state, plan, 1),
        "local", _v152_raw_info(state, plan, 2),
        "smem", _v152_raw_info(state, plan, 3),
        "bytes", plan["lay"].numel(), "valid", ok, flush=True,
    )
    return ok


def prepare(weights):
    state = _v152_v150_prepare(weights)
    state["v152_raw_enabled"] = False
    try:
        dev = weights["pooler"].device
        props = torch.cuda.get_device_properties(dev)
        vstate = state.get("v150")
        c12 = None if vstate is None else vstate.get("plans", {{}}).get("pooler")
        if props.major != 9 or "H200" not in props.name or c12 is None:
            return state

        weight = weights["pooler"]
        lay = _v152_raw_pack(weight)
        if lay is None or not torch.equal(
            _v152_raw_unpack(lay, weight.shape[0], weight.shape[1]), weight
        ):
            print("v152 raw rejected pooler pack mismatch", flush=True)
            return state
        plan = {{
            "lay": lay, "tab": c12["tab"], "esc": c12["esc"],
            "esc2": c12["esc2"], "cnt": c12["cnt"], "KS": c12["KS"],
            "n": c12["n"],
        }}
        if not _v152_raw_validate(vstate, plan, weight):
            return state

        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(152129)
        x = torch.randn((128, weight.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
        raw_out = torch.empty((128, weight.shape[0]), dtype=torch.bfloat16, device=dev)
        c12_out = torch.empty_like(raw_out)
        ratio, raw_us, c12_us = _v150_pair(
            lambda: _v152_raw_call(vstate, plan, x, raw_out),
            lambda: _v150_call(vstate, c12, x, c12_out),
            flush,
        )
        keep = raw_us < c12_us and ratio <= _V152_ROUTE_RATIO
        print(
            "v152 raw pooler candidate", raw_us, "v150_c12", c12_us,
            "core_ratio", ratio, "byte_ratio", lay.numel() / c12["lay"].numel(),
            "qualified", keep, flush=True,
        )
        if keep:
            vstate["raw_pooler"] = plan
            state["v152_raw_enabled"] = True
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        print("v152 raw unavailable", type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (state.get("v152_raw_enabled") and cfg["m"] == 128 and
            cfg["layer"] == "pooler"):
        plan = state["v150"].get("raw_pooler")
        if plan is not None:
            return _v152_raw_call(state["v150"], plan, data["x"])
    return _v152_v150_kernel_fn(data, state)
'''


base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("raw_flags", RAW_FLAGS, hex(RAW_FLAGS))
print("raw_cuda_sha256", hashlib.sha256(raw_cuda.encode()).hexdigest().upper())
print("bytes", len(candidate.encode()))
