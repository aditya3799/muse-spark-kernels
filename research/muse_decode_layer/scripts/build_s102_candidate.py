import re
import os

# Read base S081
with open('submission_s081_keell0renz_890809c.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Prune _mf_ab diagnostic lines (lines 16223-16345 in base)
mfab_marker = "# ---------------------------------------------------------------- MFIN: paired in-prepare timing of the MoE kernel (moe-fin, E300+)"
idx_mfab = src.find(mfab_marker)
if idx_mfab != -1:
    src_clean = src[:idx_mfab]
else:
    raise ValueError("MFAB marker not found!")

# 2. Modify _make_chen inside src_clean:
# 2a. Add small-m variants into RQPROD
rqprod_old = 'RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",\n              "d2s5hzwP", "s4s5hzw"]'
rqprod_new = 'RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",\n              "d2s5hzwP", "s4s5hzw", "t16s5hzw", "t16s4hzw", "t32s6hzw", "t32s5hzw", "h16s5hze2", "t16s5hze", "h32s5hze2"]'
assert rqprod_old in src_clean, "RQPROD not found"
src_clean = src_clean.replace(rqprod_old, rqprod_new, 1)

# 2b. Add small-m into RQBAKE_L
rqbake_old = '''    RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }'''

rqbake_new = '''    RQBAKE_L = {
        ("qkv_gate", 16): ["t16s5hzw", "t16s4hzw"],
        ("qkv_gate", 32): ["t32s6hzw", "t32s5hzw"],
        ("o", 16): ["t16s5hze", "t16s5hzw"],
        ("o", 32): ["t32s5hzw", "h32s5hze2"],
        ("unpooler", 16): ["t16s5hze", "t16s4hze"],
        ("unpooler", 32): ["t32s6hzw", "t32s5hzw"],
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }'''
assert rqbake_old in src_clean, "RQBAKE_L not found"
src_clean = src_clean.replace(rqbake_old, rqbake_new, 1)

# 2c. Add 16 and 32 to RQBAKE
rqbake_dict_old = 'RQBAKE = {128: [], 256: []}'
rqbake_dict_new = 'RQBAKE = {16: [], 32: [], 128: [], 256: []}'
assert rqbake_dict_old in src_clean, "rqbake_dict_old not found"
src_clean = src_clean.replace(rqbake_dict_old, rqbake_dict_new, 1)

# 2d. Add "o" to RQLAYERS
rqlayers_old = 'RQLAYERS = ("qkv_gate", "unpooler")'
rqlayers_new = 'RQLAYERS = ("qkv_gate", "o", "unpooler")'
assert rqlayers_old in src_clean, "RQLAYERS not found"
src_clean = src_clean.replace(rqlayers_old, rqlayers_new, 1)

# 2e. In _rqbake: bl_old, xs_old, m_loop_old, names_m_old
bl_old = '_bl = RQBAKE_L.get((layer, 128), RQBAKE[128]) + RQBAKE_L.get((layer, 256), RQBAKE[256])'
bl_new = '_bl = sum([RQBAKE_L.get((layer, m_), []) for m_ in (16, 32, 128, 256)], [])'
assert bl_old in src_clean, "bl_old not found"
src_clean = src_clean.replace(bl_old, bl_new, 1)

xs_old = 'for m in (128, 256):  # r6 timed before the pack (critic F1: the first r6 timing after the pack read ~4 us high)'
xs_new = 'for m in (16, 32, 128, 256):'
assert xs_old in src_clean, "xs_old not found"
src_clean = src_clean.replace(xs_old, xs_new, 1)

m_loop_old = '            for m in (128, 256):\n                x = xs[m]'
m_loop_new = '            for m in (16, 32, 128, 256):\n                x = xs[m]'
assert m_loop_old in src_clean, "m_loop_old not found"
src_clean = src_clean.replace(m_loop_old, m_loop_new, 1)

names_m_old = 'names_m = RQBAKE_L.get((layer, m), RQBAKE[m])'
names_m_new = 'names_m = RQBAKE_L.get((layer, m), RQBAKE.get(m, []))'
assert names_m_old in src_clean, "names_m_old not found"
src_clean = src_clean.replace(names_m_old, names_m_new, 1)

# 2f. Update cs in prepare to include o_raw
cs_old = 'cs = {"gv": {}, "lt": {}, "pooler_raw": weights["pooler"], "qkv_gate_raw": weights["qkv_gate"],\n              "unpooler_raw": weights["unpooler"]}'
cs_new = 'cs = {"gv": {}, "lt": {}, "pooler_raw": weights["pooler"], "qkv_gate_raw": weights["qkv_gate"],\n              "o_raw": weights["o"], "unpooler_raw": weights["unpooler"]}'
assert cs_old in src_clean, "cs_old not found"
src_clean = src_clean.replace(cs_old, cs_new, 1)

# 2g. Update _RQ_NAMES
rq_names_old = '_RQ_NAMES = {"qkv_gate": "qkv", "unpooler": "un"}'
rq_names_new = '_RQ_NAMES = {"qkv_gate": "qkv", "o": "o", "unpooler": "un"}'
assert rq_names_old in src_clean, "_RQ_NAMES not found"
src_clean = src_clean.replace(rq_names_old, rq_names_new, 1)

# 2h. Update RQ_USE
rq_use_old = 'RQ_USE = {(\'un\', 128), (\'qkv\', 256), (\'qkv\', 128)}'
rq_use_new = 'RQ_USE = {(\'un\', 128), (\'qkv\', 256), (\'qkv\', 128), (\'qkv\', 16), (\'qkv\', 32), (\'o\', 16), (\'o\', 32), (\'un\', 16), (\'un\', 32)}'
assert rq_use_old in src_clean, "RQ_USE not found"
src_clean = src_clean.replace(rq_use_old, rq_use_new, 1)

# 2i. Update _rq_gemm to handle 3D or 2D x cleanly
rq_gemm_old = '''def _rq_gemm(state, name, x, T):
    p = state.get("chen_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    if _CHEN["_rqcall"](state["chen_st"], p, x.contiguous(), out) != 0:
        return None
    return out'''

rq_gemm_new = '''def _rq_gemm(state, name, x, T):
    p = state.get("chen_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    x2d = x.reshape(T, -1).contiguous()
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    if _CHEN["_rqcall"](state["chen_st"], p, x2d, out) != 0:
        return None
    return out'''
assert rq_gemm_old in src_clean, "rq_gemm_old not found"
src_clean = src_clean.replace(rq_gemm_old, rq_gemm_new, 1)

# 3. Update dense to check _s073_cg_run and _rq_gemm
dense_old = '''def dense(x, w, T, state=None, name=None):
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)'''

dense_new = '''def dense(x, w, T, state=None, name=None):
    if state is not None:
        if T == 1 and name in state.get("s073_cg_enabled", ()):
            expected = _S073_CG_CONFIG.get(name)
            if expected is not None and (w.shape[0], w.shape[1]) == expected[:2] and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
                return _s073_cg_run(state, name, x.reshape(1, -1)).view(1, 1, expected[0]), 1
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)'''
assert dense_old in src_clean, "dense_old not found"
src_clean = src_clean.replace(dense_old, dense_new, 1)

# 4. Update kernel_fn to check _rq_gemm for T in (16, 32)
kfn_old = '''def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None'''

kfn_new = '''def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    if T in (16, 32):
        rq = _rq_gemm(state, "qkv", data["attn_input"], T)
        if rq is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg):
            try:
                o_in = _decode_attention7(data, state, rq)
            except RuntimeError:
                return _kernel_fn_fa3(data, state)
            return _after_attention(data, state, o_in, None, T)
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None'''
assert kfn_old in src_clean, "kfn_old not found"
src_clean = src_clean.replace(kfn_old, kfn_new, 1)

# 5. Append S073 C12 GEMV with "o" enabled
s073_cgemv_code = r'''
# ---------------------------------------------------------------------------
# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV, O, and sliced unpool.
# ---------------------------------------------------------------------------
_S073_CG_CUDA = '\n#include <ATen/core/Tensor.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {\n  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;\n  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);\n  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);\n}\n__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {\n  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);\n}\ntemplate <int ROWS, int NTH, int K>\n__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,\n    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,\n    __nv_bfloat16* __restrict__ out, uint32_t base4) {\n  constexpr int PER = K / (16 * NTH);\n  constexpr int NWARP = NTH / 32;\n  static_assert(PER >= 1, "cgemv: K >= 16 NTH");\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int r0 = blockIdx.x * ROWS;\n  __shared__ float red[ROWS][NWARP + 1];\n  if (warp == NWARP) {\n    const int2* tb = etab + (size_t)blockIdx.x * pmax;\n    int2 mine = make_int2(0, 0);\n    if (lane < pmax) mine = __ldg(&tb[lane]);\n    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);\n    float ec[ROWS];\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;\n    for (int i0 = 0; i0 <= cnt; i0 += 32) {\n      const int i = i0 + lane;\n      int2 en = mine;\n      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);\n      if (i >= 1 && i <= cnt) {\n        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);\n        const int rl = en.x >> 16;\n#pragma unroll\n        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;\n      }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      float v = ec[r];\n#pragma unroll\n      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n      if (lane == 0) red[r][NWARP] = v;\n    }\n    __syncthreads();\n    return;\n  }\n  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n    for (int j = 0; j < PER; ++j) {\n      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;\n      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));\n      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));\n    }\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;\n    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));\n    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));\n  }\n  float acc[ROWS];\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;\n#pragma unroll\n  for (int j = 0; j < PER; ++j) {\n    float xf[16];\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }\n    }\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r) {\n      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};\n      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};\n      float v = acc[r];\n#pragma unroll\n      for (int g = 0; g < 2; ++g) {\n        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);\n        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);\n        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];\n        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);\n        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);\n        const float* xg = xf + 8 * g;\n        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);\n        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);\n        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);\n        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);\n      }\n      acc[r] = v;\n    }\n  }\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) {\n    float v = acc[r];\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n    if (lane == 0) red[r][warp] = v;\n  }\n  __syncthreads();\n  if (tid < ROWS) {\n    float v = 0.f;\n#pragma unroll\n    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];\n    out[r0 + tid] = __float2bfloat16_rn(v);\n  }\n}\n\nvoid cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,\n           int64_t rows, int64_t nth) {\n  const int N = out.size(1), K = L.size(1);\n  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");\n  auto st = at::cuda::getCurrentCUDAStream();\n  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();\n  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());\n  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());\n  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());\n  const uint32_t b4 = (uint32_t)base4;\n  const int pm = (int)pmax;\n  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);\n  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);\n  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);\n  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);\n  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n\n'
_S073_CG_CPP = r"""
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x,
           torch::Tensor etab, int64_t pmax, torch::Tensor out,
           int64_t base4, int64_t rows, int64_t nth);
"""
_S073_CG_CONFIG = {
    "qkv": (2304, 8192, 2, 512),
    "o": (8192, 1024, 2, 64),
    "un": (1024, 4096, 2, 128),
}
_s073_cg_module_cache = [None]


def _s073_cg_module():
    if _s073_cg_module_cache[0] is None:
        from torch.utils.cpp_extension import load_inline as _s073_load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
        tag = _hashlib.sha256(_S073_CG_CUDA.encode()).hexdigest()[:16]
        _s073_cg_module_cache[0] = _s073_load_inline(
            name=f"dl_s073_cgemv_{tag}_{arch}",
            cpp_sources="#include <torch/extension.h>\n" + _S073_CG_CPP,
            cuda_sources=_S073_CG_CUDA,
            functions=["cgemv"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            no_implicit_headers=True,
            verbose=False,
        )
    return _s073_cg_module_cache[0]


def _s073_cg_compress(w, rows_per_cta):
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}


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


_s102_prior_bfp_gemm = _bfp_gemm
def _bfp_gemm(st, name, x, m, n, k):
    if m == 1 and name in st.get("s073_cg_enabled", ()):
        expected = _S073_CG_CONFIG.get(name)
        if (expected is not None and (n, k) == expected[:2] and
                x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and x.data_ptr() % 16 == 0):
            return _s073_cg_run(st, name, x)
    rq = _rq_gemm(st, name, x, m)
    if rq is not None:
        return rq
    return _s102_prior_bfp_gemm(st, name, x, m, n, k)


_s102_prior_prepare = prepare
def prepare(weights):
    st = _s102_prior_prepare(weights)
    wq = weights.get("qkv_gate")
    if (wq is None or not wq.is_cuda or torch.cuda.get_device_capability(wq.device) != (9, 0)
            or "H200" not in torch.cuda.get_device_name(wq.device)):
        return st
    st["s073_cg"] = {}
    st["s073_cg_enabled"] = set()
    try:
        _s073_cg_module()
        targets = (("qkv", weights["qkv_gate"]),
                   ("o", weights["o"]),
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

full_src = src_clean + s073_cgemv_code
out_path = 'submission_s102_rq_cgemv.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(full_src)

print(f"Generated {out_path}: {len(full_src.encode('utf-8')):,} bytes")
print("Under 1,000,000 bytes:", len(full_src.encode('utf-8')) < 1000000)
