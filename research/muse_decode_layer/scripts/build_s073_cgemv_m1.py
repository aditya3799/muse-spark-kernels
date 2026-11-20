"""Build S073 from protected S052: graft BF16 v105 exact-C12 m=1 GEMV.

The generated candidate is self-contained.  This builder reads the protected
decoder and the local component source only to copy the proven cgemv CUDA and
Python packer verbatim; the resulting submission has no parent-repository
dependency.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission.py"
COMPONENT = ROOT.parent / "muse_bf16_proj" / "submission_cgemv_threads_v105.py"
OUT = ROOT / "submission_s073_cgemv_m1.py"

BASE_SHA256 = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"
COMPONENT_SHA256 = "b703ebf929e9958718e0c44ecc17b2825406bbfb9e190012a8c7ace3782cccd5"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_assignment(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{name} is not a string literal")
            return value
    raise RuntimeError(f"missing literal assignment: {name}")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            text = ast.get_source_segment(source, node)
            if text is None:
                break
            return text
    raise RuntimeError(f"missing function: {name}")


if _sha(BASE) != BASE_SHA256:
    raise RuntimeError("protected S052 hash mismatch; refusing to generate")
if _sha(COMPONENT) != COMPONENT_SHA256:
    raise RuntimeError("protected BF16 v105 hash mismatch; refusing to copy")

base = BASE.read_text(encoding="utf-8")
component = COMPONENT.read_text(encoding="utf-8")

rgemv_src = _literal_assignment(component, "RGEMV_SRC")
cg_begin = rgemv_src.index("// ============================================================================ cgemv (m = 1)")
cg_end = rgemv_src.index("// BF16 GEMV (the unpooler slice at m = 1", cg_begin)
cg_cuda_body = rgemv_src[cg_begin:cg_end]

cg_cuda = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Copied from protected bf16_proj v105.  The 12-bit main stream clamps the
// exponent into an eight-value window; rare escapes restore the exact BF16.
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
""" + cg_cuda_body

cg_compress = _function_source(component, "cg_compress").replace(
    "def cg_compress(", "def _s073_cg_compress(", 1
)

appendix = f'''

# ---------------------------------------------------------------------------
# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool.
# Component source SHA256: {COMPONENT_SHA256}
# ---------------------------------------------------------------------------
_S073_CG_CUDA = {cg_cuda!r}
_S073_CG_CPP = r"""
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x,
           torch::Tensor etab, int64_t pmax, torch::Tensor out,
           int64_t base4, int64_t rows, int64_t nth);
"""
_S073_CG_CONFIG = {{"qkv": (2304, 8192, 2, 512),
                   "un": (1024, 4096, 2, 128)}}
_s073_cg_module_cache = [None]


def _s073_cg_module():
    if _s073_cg_module_cache[0] is None:
        # Keep the protected component's compiler/loader.  In particular,
        # no_implicit_headers is a torch cpp_extension option, not tvm_ffi.
        from torch.utils.cpp_extension import load_inline as _s073_load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{{major}}{{minor}}a" if major >= 9 else f"{{major}}{{minor}}"
        tag = _hashlib.sha256(_S073_CG_CUDA.encode()).hexdigest()[:16]
        _s073_cg_module_cache[0] = _s073_load_inline(
            name=f"dl_s073_cgemv_{{tag}}_{{arch}}",
            cpp_sources="#include <torch/extension.h>\\n" + _S073_CG_CPP,
            cuda_sources=_S073_CG_CUDA,
            functions=["cgemv"],
            # Do not force C++17: the current judge's Torch headers require
            # the loader-selected C++20 mode.  The protected v105 component
            # likewise supplies no explicit -std flag here.
            extra_cuda_cflags=["-O3",
                               f"-gencode=arch=compute_{{arch}},code=sm_{{arch}}"],
            no_implicit_headers=True,
            verbose=False,
        )
    return _s073_cg_module_cache[0]


{cg_compress}


def _s073_cg_run(state, name, x):
    n, _k, rows, nth = _S073_CG_CONFIG[name]
    cg = state["s073_cg"][name]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _s073_cg_module().cgemv(cg["L"], cg["C"], x, cg["tab"],
                            cg["pmax"], out, cg["base4"], rows, nth)
    return out


def _s073_cg_validate(state, name, weight):
    n, k, _rows, _nth = _S073_CG_CONFIG[name]
    dev = weight.device
    gen = torch.Generator(device=dev).manual_seed(73001 + (0 if name == "qkv" else 1))
    x = torch.randn((1, k), generator=gen, dtype=torch.float32, device=dev).to(torch.bfloat16)
    x[0, 0] = 0
    x[0, k - 1] = 16
    got = _s073_cg_run(state, name, x)
    ref = torch.mm(x, weight[:n].t())
    rel = float(((got.float() - ref.float()).norm() / ref.float().norm()).item())
    exact = True
    one = torch.zeros((1, k), dtype=torch.bfloat16, device=dev)
    for col in (0, k // 3, (2 * k) // 3, k - 1):
        one.zero_()
        one[0, col] = 1
        exact = exact and bool(torch.equal(_s073_cg_run(state, name, one)[0], weight[:n, col]))
    torch.cuda.synchronize(dev)
    return exact, rel, bool(torch.isfinite(got).all())


_s073_prior_bfp_gemm = _bfp_gemm
def _bfp_gemm(st, name, x, m, n, k):
    enabled = st.get("s073_cg_enabled", ())
    expected = _S073_CG_CONFIG.get(name)
    if (m == 1 and name in enabled and expected is not None and
            (n, k) == expected[:2] and x.dtype == torch.bfloat16 and
            x.is_cuda and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return _s073_cg_run(st, name, x)
    return _s073_prior_bfp_gemm(st, name, x, m, n, k)


_s073_prior_prepare = prepare
def prepare(weights):
    st = _s073_prior_prepare(weights)
    wq = weights.get("qkv_gate")
    if (wq is None or not wq.is_cuda or torch.cuda.get_device_capability(wq.device) != (9, 0)
            or "H200" not in torch.cuda.get_device_name(wq.device)):
        return st
    st["s073_cg"] = {{}}
    st["s073_cg_enabled"] = set()
    try:
        _s073_cg_module()
        targets = (("qkv", weights["qkv_gate"]),
                   ("un", weights["unpooler"][:1024].contiguous()))
        for name, weight in targets:
            rows = _S073_CG_CONFIG[name][2]
            packed = _s073_cg_compress(weight, rows)
            if packed is None:
                print("s073 cgemv rejected", name, "pack unavailable", flush=True)
                continue
            st["s073_cg"][name] = packed
            exact, rel, finite = _s073_cg_validate(st, name, weight)
            valid = exact and finite and rel < 5e-3
            if valid:
                st["s073_cg_enabled"].add(name)
            print("s073 cgemv", name, "exact", exact, "finite", finite,
                  "fp32_rel", rel, "enabled", valid, flush=True)
    except Exception as exc:
        st["s073_cg_enabled"] = set()
        print("s073 cgemv unavailable", type(exc).__name__, str(exc), flush=True)
    return st
'''

candidate = base.rstrip() + "\n\n" + appendix.lstrip()
compile(candidate, str(OUT), "exec")
OUT.write_bytes(candidate.encode())
print(OUT)
print(hashlib.sha256(candidate.encode()).hexdigest())
