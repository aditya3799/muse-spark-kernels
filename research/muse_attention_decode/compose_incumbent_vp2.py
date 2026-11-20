"""Mechanically graft the attributed exact-VP2 extension onto the scored incumbent."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
INCUMBENT = ROOT / "submission.py"
PUBLIC = ROOT / "submission_keell0renz_exact_l264.py"
OUTPUT = ROOT / "submission_incumbent_exact_vp2.py"

inc = INCUMBENT.read_text()
pub = PUBLIC.read_text()

old_entry = "def kernel_fn(data: Inputs) -> torch.Tensor:\n"
if inc.count(old_entry) != 1:
    raise RuntimeError("incumbent kernel_fn marker changed")
inc = inc.replace(old_entry, "def _incumbent_kernel_fn(data: Inputs) -> torch.Tensor:\n", 1)
inc = inc.replace(
    "# okc: problem=muse_attention_decode\n",
    "# okc: problem=muse_attention_decode\n"
    "# Exact VP2 verify-B1 sliding kernel from ZhimaoL/sglang okc_patch/_cuda_src_vfg.py (Apache-2.0).\n"
    "# The scored incumbent remains the path for every other shape.\n",
    1,
)

vfg_start = pub.index('ZH_VFG_CUDA = r"""')
cpp_start = pub.index('ZH_VFG_CPP = r"""', vfg_start)
glue_start = pub.index("_ZH = {}", cpp_start)
vfg_source = pub[vfg_start:cpp_start]
vfg_cpp = pub[cpp_start:glue_start]

glue = r'''
import hashlib as _vp2_hashlib
import threading as _vp2_threading
from torch.utils.cpp_extension import load_inline as _vp2_load_inline

_VP2 = {}
_VP2_THREADS = []


def _vp2_tag(*parts):
    return _vp2_hashlib.sha1("\n".join(parts).encode()).hexdigest()[:10]


def _vp2_build():
    try:
        flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-U__CUDA_NO_HALF_OPERATORS__",
                 "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                 "-U__CUDA_NO_HALF2_OPERATORS__", "-DZH_SCH=0"]
        _VP2["ext"] = _vp2_load_inline(
            name="okc_incumbent_vp2_" + _vp2_tag(ZH_VFG_CUDA, *flags),
            cpp_sources=[ZH_VFG_CPP], cuda_sources=[ZH_VFG_CUDA],
            functions=["muse_vfg_launch", "muse_vp2_launch"],
            extra_cuda_cflags=flags, verbose=False)
    except Exception as exc:  # a failed optional build leaves the scored incumbent in charge
        _VP2["err"] = repr(exc)[-1500:]
        print("vp2 build failed:", _VP2["err"])


def _vp2_start_builds():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return
    for fn in (_module, _vp2_build):
        th = _vp2_threading.Thread(target=fn, daemon=True)
        th.start()
        _VP2_THREADS.append(th)


def _vp2_join():
    while _VP2_THREADS:
        _VP2_THREADS.pop().join()


def _vp2_ok(data):
    cfg = data["config"]
    pt = data["page_table"]
    return (cfg["q_len"] == 8 and cfg["batch"] == 1 and cfg["window"] == 2048
            and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and pt.dtype == torch.int32 and pt.stride(1) == 1
            and data["cache_loc"].dtype == torch.int64 and data["seq_lens"].dtype == torch.int32
            and data["k_cache"].dtype == torch.float8_e4m3fn and data["v_cache"].dtype == torch.float8_e4m3fn
            and data["qkv"].dtype == torch.bfloat16 and data["qkv"].is_contiguous() and _fits(data))


def _vp2_launch(data):
    cfg = data["config"]
    qkv = data["qkv"]
    device = qkv.device
    pt = data["page_table"]
    out = torch.empty((1, 8, 16 * 64), dtype=torch.bfloat16, device=device)
    freq = _inv_freq(cfg["rope_theta"], device).data_ptr()
    c2 = cfg["qk_scale_factor"] * cfg["sm_scale"] * cfg["kv_scale"] * LOG2E
    rk = 1.0 / cfg["kv_scale"]
    _VP2["ext"].muse_vp2_launch(
        qkv.data_ptr(), pt.data_ptr(), pt.stride(0), data["seq_lens"].data_ptr(), data["cache_loc"].data_ptr(),
        data["k_cache"].data_ptr(), data["v_cache"].data_ptr(), freq, out.data_ptr(), 1, cfg["window"], c2,
        rk, rk, cfg["kv_scale"], 1e-5, 0, torch.cuda.current_stream().cuda_stream)
    return out


_vp2_start_builds()


def kernel_fn(data: Inputs) -> torch.Tensor:
    if _VP2_THREADS:
        _vp2_join()
    if "ext" in _VP2 and _vp2_ok(data):
        return _vp2_launch(data)
    return _incumbent_kernel_fn(data)
'''

OUTPUT.write_text(inc.rstrip() + "\n\n" + vfg_source + vfg_cpp + glue.lstrip())
print(OUTPUT)
