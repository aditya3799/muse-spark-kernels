"""Mechanically integrate the attributed exact-VP2 kernel into the incumbent extension."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
INCUMBENT = ROOT / "submission.py"
PUBLIC = ROOT / "submission_keell0renz_exact_l264.py"
OUTPUT = ROOT / "submission_incumbent_vp2_integrated.py"

inc = INCUMBENT.read_text()
pub = PUBLIC.read_text()

vfg_start = pub.index('ZH_VFG_CUDA = r"""')
cpp_start = pub.index('ZH_VFG_CPP = r"""', vfg_start)
glue_start = pub.index("_ZH = {}", cpp_start)
vfg_assignment = pub[vfg_start:cpp_start]
cpp_assignment = pub[cpp_start:glue_start]
vfg_cuda = vfg_assignment.split('r"""', 1)[1].rsplit('"""', 1)[0]
vfg_cpp = cpp_assignment.split('r"""', 1)[1].rsplit('"""', 1)[0]

inc = inc.replace(
    "# okc: problem=muse_attention_decode\n",
    "# okc: problem=muse_attention_decode\n"
    "# Exact VP2 verify-B1 sliding kernel from ZhimaoL/sglang okc_patch/_cuda_src_vfg.py (Apache-2.0).\n"
    "# It is compiled into the incumbent extension; every other shape retains the incumbent path.\n",
    1,
)
cuda_end = '\n"""\n\nCPP_SRC = r"""'
if inc.count(cuda_end) != 1:
    raise RuntimeError("CUDA_SRC terminator changed")
inc = inc.replace(cuda_end, "\n\n// ---- attributed exact VP2 extension ----\n" + vfg_cuda + cuda_end, 1)

cpp_end = '\n"""\n\n_MOD = None'
if inc.count(cpp_end) != 1:
    raise RuntimeError("CPP_SRC terminator changed")
inc = inc.replace(cpp_end, "\n" + vfg_cpp + cpp_end, 1)
inc = inc.replace('name="okc_attn_v298_integrated_90a"', 'name="okc_attn_v298_integrated_vp2_90a"', 1)
inc = inc.replace('functions=["run"]', 'functions=["run", "muse_vp2_launch"]', 1)
inc = inc.replace(
    '"--expt-relaxed-constexpr"],',
    '"--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__",\n'
    '                           "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",\n'
    '                           "-U__CUDA_NO_HALF2_OPERATORS__", "-DZH_SCH=0"],',
    1,
)

entry = "def kernel_fn(data: Inputs) -> torch.Tensor:\n"
if inc.count(entry) != 1:
    raise RuntimeError("incumbent kernel_fn marker changed")
inc = inc.replace(entry, "def _incumbent_kernel_fn(data: Inputs) -> torch.Tensor:\n", 1)

glue = r'''


def _integrated_vp2_ok(data):
    cfg = data["config"]
    pt = data["page_table"]
    return (cfg["q_len"] == 8 and cfg["batch"] == 1 and cfg["window"] == 2048
            and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and pt.dtype == torch.int32 and pt.stride(1) == 1
            and data["cache_loc"].dtype == torch.int64 and data["seq_lens"].dtype == torch.int32
            and data["k_cache"].dtype == torch.float8_e4m3fn and data["v_cache"].dtype == torch.float8_e4m3fn
            and data["qkv"].dtype == torch.bfloat16 and data["qkv"].is_contiguous() and _fits(data))


def _integrated_vp2_launch(data):
    cfg = data["config"]
    qkv = data["qkv"]
    device = qkv.device
    pt = data["page_table"]
    out = torch.empty((1, 8, 16 * 64), dtype=torch.bfloat16, device=device)
    freq = _inv_freq(cfg["rope_theta"], device).data_ptr()
    c2 = cfg["qk_scale_factor"] * cfg["sm_scale"] * cfg["kv_scale"] * LOG2E
    rk = 1.0 / cfg["kv_scale"]
    _module().muse_vp2_launch(
        qkv.data_ptr(), pt.data_ptr(), pt.stride(0), data["seq_lens"].data_ptr(), data["cache_loc"].data_ptr(),
        data["k_cache"].data_ptr(), data["v_cache"].data_ptr(), freq, out.data_ptr(), 1, cfg["window"], c2,
        rk, rk, cfg["kv_scale"], 1e-5, 0, torch.cuda.current_stream().cuda_stream)
    return out


def kernel_fn(data: Inputs) -> torch.Tensor:
    if _integrated_vp2_ok(data):
        return _integrated_vp2_launch(data)
    return _incumbent_kernel_fn(data)
'''

OUTPUT.write_text(inc.rstrip() + glue)
print(OUTPUT)
