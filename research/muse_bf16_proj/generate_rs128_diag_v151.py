"""Generate v151: diagnostic-only cycle instrumentation for v150 RS128.

The scored v150 file remains byte-identical. This candidate adds the upstream
INS=1 twin of the selected RS128 specialization and invokes it only from
prepare() for pooler m=128. kernel_fn and production dispatch remain v150.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import generate_rs128_faithful_v150 as v150


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_diag_v151.py"

diag_case = (
    v150.case
    + "\n    case 1: return rq_go<128, 5, 1, 0, 0, 1, 42999939, 1>"
      "(p, x, out, (int)KS, (int)info);"
)
diag_cuda = v150.cuda_source.replace(v150.case, diag_case)
if diag_cuda == v150.cuda_source:
    raise RuntimeError("failed to add INS=1 RS128 specialization")
diag_payload = base64.b64encode(zlib.compress(diag_cuda.encode(), 9)).decode()

append = f'''

# ---------------------------------------------------------------- v151: diagnostic-only RS128 cycle attribution.
# Production case 0 and kernel_fn are unchanged from v150. Case 1 is the
# upstream INS=1 twin and is called only by prepare() for pooler m=128.
_V150_CUDA_PAYLOAD = "{diag_payload}"
_V151_CUDA_SHA256 = "{hashlib.sha256(diag_cuda.encode()).hexdigest().upper()}"


def _v151_build():
    if _v150_module[0] is None:
        src = _v150_cuda_source()
        tag = "v151_rs128_diag_" + _v150_hashlib.sha256(src.encode()).hexdigest()[:16]
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


_v150_build = _v151_build
_v151_production_prepare = prepare


def _v151_middle(values):
    values = sorted(int(v) for v in values)
    return values[len(values) // 2]


def _v151_pooler_report(state, weights):
    vstate = state.get("v150")
    if not vstate:
        print("v151 pooler128 diagnostic unavailable: no v150 state", flush=True)
        return
    plan = vstate.get("plans", {{}}).get("pooler")
    if plan is None:
        print("v151 pooler128 diagnostic unavailable: no packed plan", flush=True)
        return

    weight = weights["pooler"]
    ks = int(plan["KS"])
    nb = int(plan["lay"].shape[0])
    ncta = ks * nb
    # rqrun requires room for both possible 64-row halves even though this
    # full-width specialization writes only the first ncta rows.
    prof = torch.zeros(2 * ncta * 32, dtype=torch.int64, device=weight.device)
    flush = torch.empty((16384, 1024), dtype=torch.float32, device=weight.device)
    flush_sink = torch.empty(1024, dtype=torch.float32, device=weight.device)
    gen = torch.Generator(device=weight.device).manual_seed(151128)
    x = torch.randn(
        (128, weight.shape[1]), generator=gen, device=weight.device
    ).to(torch.bfloat16)
    out = torch.empty(
        (128, weight.shape[0]), dtype=torch.bfloat16, device=weight.device
    )

    names = (
        "main_total", "full_wait_lds", "wslot_release", "c12_decode",
        "lds_data_wait", "wgmma_issue", "wgmma_wait_fix", "xslot_release",
        "epilogue",
    )
    values = {{name: [] for name in names}}
    values.update({{
        "w_producer_backpressure": [], "x_producer_backpressure": [],
        "global_prologue_ns": [], "global_main_ns": [], "global_epilogue_ns": [],
        "global_total_ns": [], "first_wgmma_ns": [], "w_producer_done_ns": [],
        "epi_wg_barrier": [], "epi_stmatrix_barrier": [], "epi_zero_wait": [],
        "epi_reduce_issue": [], "epi_reduce_wait": [],
    }})
    max_rel = 0.0

    for _ in range(5):
        torch.sum(flush, dim=0, out=flush_sink)
        plan["cnt"].zero_()
        prof.zero_()
        rc = vstate["ext"].rqrun(
            1, plan["lay"], x, out, plan["esc"], plan["esc2"], plan["cnt"],
            ks, plan["tab"], 0, prof, vstate["sink"], 0,
        )
        if rc != 0:
            raise RuntimeError("v151 diagnostic rqrun error %d" % rc)
        torch.cuda.synchronize()
        ref = x[:8].double() @ weight.double().T
        rel = float((out[:8].double() - ref).norm() / ref.norm())
        max_rel = max(max_rel, rel)
        rows = prof[: ncta * 32].view(ncta, 32).cpu()
        for i, name in enumerate(names):
            values[name].extend(rows[:, i].tolist())
            values[name].extend(rows[:, 16 + i].tolist())
        values["w_producer_backpressure"].extend(rows[:, 9].tolist())
        values["x_producer_backpressure"].extend(rows[:, 10].tolist())
        values["global_prologue_ns"].extend((rows[:, 12] - rows[:, 11]).tolist())
        values["global_main_ns"].extend((rows[:, 13] - rows[:, 12]).tolist())
        values["global_epilogue_ns"].extend((rows[:, 14] - rows[:, 13]).tolist())
        values["global_total_ns"].extend((rows[:, 14] - rows[:, 11]).tolist())
        values["first_wgmma_ns"].extend((rows[:, 25] - rows[:, 11]).tolist())
        values["w_producer_done_ns"].extend((rows[:, 15] - rows[:, 11]).tolist())
        ce = rows[:, 26:31]
        values["epi_wg_barrier"].extend(ce[:, 0].tolist())
        values["epi_stmatrix_barrier"].extend((ce[:, 1] - ce[:, 0]).tolist())
        values["epi_zero_wait"].extend((ce[:, 2] - ce[:, 1]).tolist())
        values["epi_reduce_issue"].extend((ce[:, 3] - ce[:, 2]).tolist())
        values["epi_reduce_wait"].extend((ce[:, 4] - ce[:, 3]).tolist())

    med = {{name: _v151_middle(sample) for name, sample in values.items()}}
    main = max(med["main_total"], 1)
    parts = (
        "full_wait_lds", "wslot_release", "c12_decode", "lds_data_wait",
        "wgmma_issue", "wgmma_wait_fix", "xslot_release",
    )
    print(
        "v151 pooler128 diagnostic", "ctas", ncta, "repeats", 5,
        "fp64", max_rel, "counter_sum", int(plan["cnt"].abs().sum()), flush=True,
    )
    print(
        "v151 pooler128 cycles", "main", main,
        *sum(([name, med[name]] for name in parts), []),
        "epilogue", med["epilogue"], flush=True,
    )
    print(
        "v151 pooler128 main_share",
        *sum(([name, round(med[name] / main, 4)] for name in parts), []),
        "part_sum", round(sum(med[name] for name in parts) / main, 4), flush=True,
    )
    print(
        "v151 pooler128 producers",
        "w_backpressure", med["w_producer_backpressure"],
        "x_backpressure", med["x_producer_backpressure"], flush=True,
    )
    print(
        "v151 pooler128 global_ns",
        "prologue", med["global_prologue_ns"], "main", med["global_main_ns"],
        "epilogue", med["global_epilogue_ns"], "total", med["global_total_ns"],
        "first_wgmma", med["first_wgmma_ns"],
        "w_producer_done", med["w_producer_done_ns"], flush=True,
    )
    print(
        "v151 pooler128 epilogue_cycles",
        "wg_barrier", med["epi_wg_barrier"],
        "stmatrix_barrier", med["epi_stmatrix_barrier"],
        "zero_wait", med["epi_zero_wait"],
        "reduce_issue", med["epi_reduce_issue"],
        "reduce_wait", med["epi_reduce_wait"], flush=True,
    )


def prepare(weights):
    state = _v151_production_prepare(weights)
    try:
        props = torch.cuda.get_device_properties(weights["pooler"].device)
        if props.major == 9 and "H200" in props.name:
            _v151_pooler_report(state, weights)
    except Exception as exc:
        print(
            "v151 pooler128 diagnostic failed",
            type(exc).__name__, str(exc), flush=True,
        )
    return state
'''

base = BASE.read_text()
candidate = base + append
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print("v150_sha256", hashlib.sha256(BASE.read_bytes()).hexdigest().upper())
print("diag_cuda_sha256", hashlib.sha256(diag_cuda.encode()).hexdigest().upper())
print("bytes", len(candidate.encode()))
