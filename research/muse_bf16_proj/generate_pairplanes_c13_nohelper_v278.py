"""Generate v278: exact C13 pair planes with no dedicated escape warp."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
OUTPUT = ROOT / "submission_plain_pairplanes_c13_nohelper_v278.py"

text = BASE.read_text(encoding="utf-8").rstrip()

block = r'''


# v278: row-paired C13 GEMV for m=1.  Sign and four exponent-offset bits
# cover a 16-exponent window; the challenge weights leave only a handful of
# whole-matrix exceptions.  The two final reducer threads restore those exact
# FP32 deltas, so the compute block needs no helper warp or correction barrier.
_V278_SHAPES = {
    'qkv_gate': (2304, 8192, 512),
    'o': (8192, 1024, 64),
    'pooler': (4096, 8192, 512),
    'unpooler': (1024, 4096, 128),
}
_V278_REPLAY_OVERHEAD_US = 4.0
_V278_REQUIRED_CORE_RATIO = 0.99

_V278_CUDA = r"""\
__device__ __forceinline__ uint32_t v278_hib13(
    uint32_t c, uint32_t signs, uint32_t base4) {
  const uint32_t spread = 0x10204080u;
  const uint32_t sb = ((signs & 0xFu) * spread) & 0x80808080u;
  return ((c & 0x0F0F0F0Fu) + base4) | sb;
}

template <int NTH, int K>
__global__ void __launch_bounds__(NTH) cgemv_pair13_kernel(
    const uint8_t* __restrict__ P, const __nv_bfloat16* __restrict__ x,
    const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int ROWS = 2;
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  constexpr int CHUNKS = K / 16;
  constexpr int PAIR_BYTES = 52 * CHUNKS;
  static_assert(PER >= 1, "cgemv_pair13: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[ROWS][NWARP];

  uint4 lw[ROWS][PER];
  uint4 cw[PER];
  uint32_t sw[PER];
  uint4 xv[PER][2];
  const uint8_t* pb = P + (size_t)blockIdx.x * PAIR_BYTES;
  const uint4* pv = reinterpret_cast<const uint4*>(pb);
  const uint32_t* ps = reinterpret_cast<const uint32_t*>(
      pb + 48 * CHUNKS);
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int q = j * NTH + tid;
    lw[0][j] = __ldg(pv + q);
    lw[1][j] = __ldg(pv + CHUNKS + q);
    cw[j] = __ldg(pv + 2 * CHUNKS + q);
    sw[j] = __ldg(ps + q);
  }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }

  float acc[ROWS] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16);
        xf[8 * h + 2 * e + 1] =
            __uint_as_float(xw[e] & 0xFFFF0000u);
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {
          lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {
          r == 0 ? cw[j].x : cw[j].z,
          r == 0 ? cw[j].y : cw[j].w};
      const uint32_t signs16 = r == 0 ? (sw[j] & 0xFFFFu) : (sw[j] >> 16);
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const uint32_t codes = cws[g];
        const uint32_t signs8 = (signs16 >> (8 * g)) & 0xFFu;
        const uint32_t ha = v278_hib13(
            codes & 0x0F0F0F0Fu, signs8 & 0xFu, base4);
        const uint32_t hb = v278_hib13(
            (codes >> 4) & 0x0F0F0F0Fu, signs8 >> 4, base4);
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140);
        const uint32_t p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140);
        const uint32_t p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v);
        v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v);
        v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v);
        v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v);
        v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    const int cnt = tb[0].x;
    float ec = 0.f;
    for (int e = 1; e <= cnt; ++e) {
      const int2 en = tb[e];
      if ((en.x >> 16) == tid) {
        ec += __bfloat162float(x[en.x & 0xFFFF]) *
              __uint_as_float((uint32_t)en.y);
      }
    }
    out[2 * blockIdx.x + tid] = __float2bfloat16_rn(v + ec);
  }
}

template <int NTH, int K>
int64_t cgemv_pair13_info_t() {
  auto kern = cgemv_pair13_kernel<NTH, K>;
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kern, NTH, 0));
  return (int64_t)attr.numRegs | ((int64_t)attr.localSizeBytes << 10) |
         ((int64_t)attr.sharedSizeBytes << 30) | ((int64_t)active << 50);
}

int64_t cgemv_pair13_info(int64_t k, int64_t nth) {
  if (k == 8192 && nth == 512) return cgemv_pair13_info_t<512,8192>();
  if (k == 1024 && nth == 64) return cgemv_pair13_info_t<64,1024>();
  if (k == 4096 && nth == 128) return cgemv_pair13_info_t<128,4096>();
  TORCH_CHECK(false, "cgemv_pair13_info: unsupported instance");
}

void cgemv_pair13(at::Tensor P, at::Tensor x, at::Tensor etab,
                  int64_t pmax, at::Tensor out, int64_t base4, int64_t nth) {
  const int N = out.size(1), K = x.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % 2 == 0,
              "cgemv_pair13: m=1, even N");
  auto st = at::cuda::getCurrentCUDAStream();
  auto pp = P.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && nth == 512)
    cgemv_pair13_kernel<512,8192><<<N/2,512,0,st>>>(pp,xp,tp,pm,op,b4);
  else if (K == 1024 && nth == 64)
    cgemv_pair13_kernel<64,1024><<<N/2,64,0,st>>>(pp,xp,tp,pm,op,b4);
  else if (K == 4096 && nth == 128)
    cgemv_pair13_kernel<128,4096><<<N/2,128,0,st>>>(pp,xp,tp,pm,op,b4);
  else TORCH_CHECK(false, "cgemv_pair13: unsupported instance");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

RGEMV_SRC += _V278_CUDA
CPP_SRC += r"""
void cgemv_pair13(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                  torch::Tensor, int64_t, int64_t);
int64_t cgemv_pair13_info(int64_t, int64_t);
"""


def _get_ext():
    """v278 translation unit with literal exact C13 pair GEMV."""
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace(
            "@@BODY@@", WG_HELPERS if sm90 else "")
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC +
                    _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v278_pair13_{arch}" +
                 (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq",
                       "gqx", "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "cgemv_pair", "cgemv_pair_info",
                       "cgemv_pair13", "cgemv_pair13_info", "rk"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                f"-gencode=arch=compute_{arch},code=sm_{arch}"
            ] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


def _v278_pack(w):
    """Create row-paired low/exponent/sign planes and exact delta tables."""
    n, k = w.shape
    if n % 2 or k % 16:
        raise RuntimeError('v278 requires even N and K divisible by 16')
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 16].sum() for b in range(113)])
    base = int(win.argmax())
    off = (e71 - base).clamp(0, 15)
    esc = (e71 < base) | (e71 > base + 15)
    chunks = k // 16
    codes8 = off.to(torch.uint8).view(n, k // 8, 8)
    codes = (codes8[:, :, 0:4] |
             (codes8[:, :, 4:8] << 4)).reshape(n, chunks, 8)
    sg = sign.view(n, chunks, 16)
    shifts = torch.arange(16, dtype=torch.int32, device=w.device)
    sb = (sg << shifts).sum(-1)
    signs = torch.stack((sb & 0xFF, (sb >> 8) & 0xFF), -1).to(torch.uint8)
    lp = low.to(torch.uint8).view(n // 2, 2, chunks, 16)
    cp = codes.view(n // 2, 2, chunks, 8)
    sp = signs.view(n // 2, 2, chunks, 2)
    pair = torch.empty((n // 2, 52 * chunks), dtype=torch.uint8,
                       device=w.device)
    pair[:, 0:16 * chunks].view(n // 2, chunks, 16).copy_(lp[:, 0])
    pair[:, 16 * chunks:32 * chunks].view(
        n // 2, chunks, 16).copy_(lp[:, 1])
    pair[:, 32 * chunks:48 * chunks].view(
        n // 2, chunks, 16)[:, :, :8].copy_(cp[:, 0])
    pair[:, 32 * chunks:48 * chunks].view(
        n // 2, chunks, 16)[:, :, 8:].copy_(cp[:, 1])
    sign_plane = pair[:, 48 * chunks:].view(n // 2, chunks, 4)
    sign_plane[:, :, :2].copy_(sp[:, 0])
    sign_plane[:, :, 2:].copy_(sp[:, 1])
    pair = pair.contiguous()
    pack_exact = (pair.numel() == n * k * 13 // 8 and
                  bool(torch.equal(
                      pair[:, :16 * chunks].view(n // 2, chunks, 16),
                      lp[:, 0])) and
                  bool(torch.equal(
                      pair[:, 16 * chunks:32 * chunks].view(
                          n // 2, chunks, 16), lp[:, 1])) and
                  bool(torch.equal(
                      pair[:, 32 * chunks:48 * chunks].view(
                          n // 2, chunks, 16)[:, :, :8], cp[:, 0])) and
                  bool(torch.equal(
                      pair[:, 32 * chunks:48 * chunks].view(
                          n // 2, chunks, 16)[:, :, 8:], cp[:, 1])) and
                  bool(torch.equal(sign_plane[:, :, :2], sp[:, 0])) and
                  bool(torch.equal(sign_plane[:, :, 2:], sp[:, 1])))

    rows, cols = esc.nonzero(as_tuple=True)
    dec_bits = ((((sign[rows, cols] << 7) |
                   (base + off[rows, cols])) << 8) |
                 low[rows, cols]) & 0xFFFF
    decoded = dec_bits.to(torch.int16).contiguous().view(
        torch.bfloat16).float()
    original = w[rows, cols].float()
    delta = (original - decoded).contiguous()
    delta_exact = bool(torch.equal(decoded + delta, original))
    cta = rows // 2
    counts = torch.bincount(cta, minlength=n // 2)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    table = torch.zeros((n // 2, pmax, 2), dtype=torch.int32,
                        device=w.device)
    table[:, 0, 0] = counts.to(torch.int32)
    table[cta, pos, 0] = (((rows - 2 * cta) << 16) | cols).to(torch.int32)
    table[cta, pos, 1] = delta.view(torch.int32)
    return {'packed': pair, 'table': table.contiguous(), 'pmax': pmax,
            'base4': base * 0x01010101, 'escapes': int(rows.numel()),
            'pack_exact': pack_exact, 'delta_exact': delta_exact}


def _v278_run(x, layer, n, state):
    entry = state['_v278_pair13'][layer]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().cgemv_pair13(
        entry['packed'], x, entry['table'], entry['pmax'], out,
        entry['base4'], entry['nth'])
    return out


def _v278_resource_info(k, nth):
    packed = int(_get_ext().cgemv_pair13_info(k, nth))
    return {
        'registers': packed & ((1 << 10) - 1),
        'local_bytes': (packed >> 10) & ((1 << 20) - 1),
        'static_shared_bytes': (packed >> 30) & ((1 << 20) - 1),
        'active_ctas_per_sm': packed >> 50,
    }


_v278_base_prepare = prepare
_v278_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v278_base_prepare(weights)
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    state['_v278_pair13'] = {}
    for li, (layer, (n, k, nth)) in enumerate(_V278_SHAPES.items()):
        try:
            entry = _v278_pack(weights[layer][:n])
            entry.update({'nth': nth, 'enabled': False})
            state['_v278_pair13'][layer] = entry
            resources = _v278_resource_info(k, nth)
            gen = torch.Generator(device=wq.device).manual_seed(278101 + li)
            x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                            device=wq.device).to(torch.bfloat16)
            x[0, 0] = 0
            x[0, k - 1] = 16
            data = {'x': x, 'config': {
                'm': 1, 'n': n, 'k': k, 'layer': layer}}
            incumbent = _v278_base_kernel_fn(data, state)
            candidate = _v278_run(x, layer, n, state)
            torch.cuda.synchronize(wq.device)
            bitwise = bool(torch.equal(candidate, incumbent))
            w = weights[layer][:n]
            ref = (x.double() @ w.double().t()).to(torch.bfloat16).double()
            rel = float(((candidate.double() - ref).norm() /
                         ref.norm()).item())
            finite = bool(torch.isfinite(candidate).all())
            probe_exact = True
            one = torch.zeros((1, k), dtype=torch.bfloat16, device=wq.device)
            for col in (0, k // 3, (2 * k) // 3, k - 1):
                one.zero_()
                one[0, col] = 1
                got = _v278_run(one, layer, n, state)
                probe_exact = (probe_exact and
                               bool(torch.equal(got[0], w[:, col])))
            torch.cuda.synchronize(wq.device)
            cand_ms, base_ms = _v102_paired_graph_ms(
                lambda: _v278_run(x, layer, n, state),
                lambda: _v278_base_kernel_fn(data, state),
                wq.device, repeats=13)
            cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
            cand_core = max(cand_us - _V278_REPLAY_OVERHEAD_US, 1e-6)
            base_core = max(base_us - _V278_REPLAY_OVERHEAD_US, 1e-6)
            core_ratio = cand_core / base_core
            valid = (entry['pack_exact'] and entry['delta_exact'] and
                     bitwise and probe_exact and finite and rel < 5e-3 and
                     resources['local_bytes'] == 0 and
                     resources['active_ctas_per_sm'] >= 1)
            enable = (valid and cand_us < base_us and
                      core_ratio <= _V278_REQUIRED_CORE_RATIO)
            entry.update({'enabled': enable, 'core_ratio': core_ratio})
            print('v278 pair13', layer, 'escapes', entry['escapes'],
                  'bytes', entry['packed'].numel(), 'resources', resources,
                  'pack_exact', entry['pack_exact'],
                  'delta_exact', entry['delta_exact'],
                  'bitwise_base', bitwise, 'probe_exact', probe_exact,
                  'finite', finite, 'fp64_rel', rel,
                  'candidate_us', cand_us, 'incumbent_us', base_us,
                  'core_ratio', core_ratio, 'enabled', enable, flush=True)
        except Exception as exc:
            print('v278 pair13 unavailable', layer, type(exc).__name__,
                  str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    entry = state.get('_v278_pair13', {}).get(cfg['layer'])
    expected = _V278_SHAPES.get(cfg['layer'])
    if (entry is not None and entry.get('enabled') and cfg['m'] == 1 and
            expected is not None and (cfg['n'], cfg['k']) == expected[:2]):
        x = data['x']
        packed = entry['packed']
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == packed.device and x.data_ptr() % 16 == 0):
            return _v278_run(x, cfg['layer'], cfg['n'], state)
    return _v278_base_kernel_fn(data, state)
'''

text += block
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
