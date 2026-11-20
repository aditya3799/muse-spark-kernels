"""Generate v272: exact pair-plane m=1 row-group tournament on v248.

Each CTA processes 1, 2, or 4 consecutive two-row tiles while retaining the
input vector in registers.  The per-output FMA and reduction order is unchanged;
the experiment only removes redundant x loads and some CTA scheduling.  Every
selected route must remain bitwise equal to the complete v248 incumbent and
beat it by at least one percent after graph-replay subtraction.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
OUTPUT = ROOT / "submission_plain_pairplanes_rowgroup_v272.py"

text = BASE.read_text(encoding="utf-8").rstrip()

block = r'''


# v272: pair-plane CTA row grouping.  Literal CUDA is compiled once in prepare;
# no source payload is compressed, encoded, or reconstructed at run time.
_V272_SHAPES = {
    'qkv_gate': (2304, 8192, 512),
    'o': (8192, 1024, 64),
    'pooler': (4096, 8192, 512),
    'unpooler': (1024, 4096, 128),
}
_V272_GROUPS = (1, 2, 4)
_V272_REPLAY_OVERHEAD_US = 4.0
_V272_REQUIRED_CORE_RATIO = 0.99

_V272_CUDA = r"""\
template <int NTH, int K, int GROUP>
__global__ void __launch_bounds__(NTH + 32) cgemv_pair_group_kernel(
    const uint8_t* __restrict__ P, const __nv_bfloat16* __restrict__ x,
    const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4, int npairs) {
  constexpr int ROWS = 2;
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  constexpr int CHUNKS = K / 16;
  static_assert(PER >= 1, "cgemv_pair_group: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[ROWS][NWARP + 1];

  uint4 xv[PER][2];
  if (warp < NWARP) {
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
      xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
      xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
    }
  }

#pragma unroll
  for (int gg = 0; gg < GROUP; ++gg) {
    const int pair = (int)blockIdx.x * GROUP + gg;
    if (pair >= npairs) break;
    if (warp == NWARP) {
      const int2* tb = etab + (size_t)pair * pmax;
      int2 mine = make_int2(0, 0);
      if (lane < pmax) mine = __ldg(&tb[lane]);
      const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
      float ec[ROWS] = {0.f, 0.f};
      for (int i0 = 0; i0 <= cnt; i0 += 32) {
        const int i = i0 + lane;
        int2 en = mine;
        if (i0 > 0)
          en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
        if (i >= 1 && i <= cnt) {
          const float c = __bfloat162float(x[en.x & 0xFFFF]) *
                          esc_delta((uint32_t)en.y, base4 & 0xFFu);
          const int rl = en.x >> 16;
          ec[0] += (rl == 0) ? c : 0.f;
          ec[1] += (rl == 1) ? c : 0.f;
        }
      }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        float v = ec[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[r][NWARP] = v;
      }
    } else {
      uint4 lw[ROWS][PER];
      uint4 cw[PER];
      const uint4* pv = reinterpret_cast<const uint4*>(P) +
                        (size_t)pair * 3 * CHUNKS;
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        const int q = j * NTH + tid;
        lw[0][j] = __ldg(pv + q);
        lw[1][j] = __ldg(pv + CHUNKS + q);
        cw[j] = __ldg(pv + 2 * CHUNKS + q);
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
          float v = acc[r];
#pragma unroll
          for (int g = 0; g < 2; ++g) {
            const uint32_t ha =
                cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
            const uint32_t hb =
                cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);
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
    }
    __syncthreads();
    if (tid < ROWS) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
      out[2 * pair + tid] = __float2bfloat16_rn(v);
    }
    __syncthreads();
  }
}

template <int NTH, int K, int GROUP>
int64_t cgemv_pair_group_info_t() {
  auto kern = cgemv_pair_group_kernel<NTH, K, GROUP>;
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kern, NTH + 32, 0));
  return (int64_t)attr.numRegs | ((int64_t)attr.localSizeBytes << 10) |
         ((int64_t)attr.sharedSizeBytes << 30) | ((int64_t)active << 50);
}

int64_t cgemv_pair_group_info(int64_t k, int64_t nth, int64_t group) {
#define V272_INFO(K0, N0, G0) \
  if (k == K0 && nth == N0 && group == G0) \
    return cgemv_pair_group_info_t<N0, K0, G0>()
  V272_INFO(8192, 512, 1); V272_INFO(8192, 512, 2);
  V272_INFO(8192, 512, 4); V272_INFO(1024, 64, 1);
  V272_INFO(1024, 64, 2); V272_INFO(1024, 64, 4);
  V272_INFO(4096, 128, 1); V272_INFO(4096, 128, 2);
  V272_INFO(4096, 128, 4);
#undef V272_INFO
  TORCH_CHECK(false, "cgemv_pair_group_info: unsupported instance");
}

void cgemv_pair_group(at::Tensor P, at::Tensor x, at::Tensor etab,
                      int64_t pmax, at::Tensor out, int64_t base4,
                      int64_t nth, int64_t group) {
  const int N = out.size(1), K = x.size(1), npairs = N / 2;
  TORCH_CHECK(x.size(0) == 1 && N % 2 == 0,
              "cgemv_pair_group: m=1, even N");
  auto st = at::cuda::getCurrentCUDAStream();
  auto pp = P.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
#define V272_LAUNCH(K0, N0, G0) \
  if (K == K0 && nth == N0 && group == G0) { \
    cgemv_pair_group_kernel<N0, K0, G0><<<(npairs + G0 - 1) / G0, \
        N0 + 32, 0, st>>>(pp, xp, tp, pm, op, b4, npairs); \
  }
  bool launched = true;
  if (K == 8192 && nth == 512 && group == 1)
    cgemv_pair_group_kernel<512,8192,1><<<npairs,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 8192 && nth == 512 && group == 2)
    cgemv_pair_group_kernel<512,8192,2><<<(npairs+1)/2,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 8192 && nth == 512 && group == 4)
    cgemv_pair_group_kernel<512,8192,4><<<(npairs+3)/4,544,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 1024 && nth == 64 && group == 1)
    cgemv_pair_group_kernel<64,1024,1><<<npairs,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 1024 && nth == 64 && group == 2)
    cgemv_pair_group_kernel<64,1024,2><<<(npairs+1)/2,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 1024 && nth == 64 && group == 4)
    cgemv_pair_group_kernel<64,1024,4><<<(npairs+3)/4,96,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 4096 && nth == 128 && group == 1)
    cgemv_pair_group_kernel<128,4096,1><<<npairs,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 4096 && nth == 128 && group == 2)
    cgemv_pair_group_kernel<128,4096,2><<<(npairs+1)/2,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else if (K == 4096 && nth == 128 && group == 4)
    cgemv_pair_group_kernel<128,4096,4><<<(npairs+3)/4,160,0,st>>>(pp,xp,tp,pm,op,b4,npairs);
  else launched = false;
#undef V272_LAUNCH
  TORCH_CHECK(launched, "cgemv_pair_group: unsupported instance");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

RGEMV_SRC += _V272_CUDA
CPP_SRC += r"""
void cgemv_pair_group(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                      torch::Tensor, int64_t, int64_t, int64_t);
int64_t cgemv_pair_group_info(int64_t, int64_t, int64_t);
"""


def _get_ext():
    """v272 translation unit with literal row-group pair-plane kernels."""
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
            name=f"okc_bf16proj_v272_pairgroup_{arch}" +
                 (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq",
                       "gqx", "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "cgemv_pair", "cgemv_pair_info",
                       "cgemv_pair_group", "cgemv_pair_group_info", "rk"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                f"-gencode=arch=compute_{arch},code=sm_{arch}"
            ] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


def _v272_run(x, layer, n, state, group):
    entry = state['_v272_pair_group'][layer]
    cg = state['cg'][layer]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().cgemv_pair_group(entry['packed'], x, cg['tab'], cg['pmax'],
                                out, cg['base4'], entry['nth'], group)
    return out


def _v272_resource_info(k, nth, group):
    packed = int(_get_ext().cgemv_pair_group_info(k, nth, group))
    return {
        'registers': packed & ((1 << 10) - 1),
        'local_bytes': (packed >> 10) & ((1 << 20) - 1),
        'static_shared_bytes': (packed >> 30) & ((1 << 20) - 1),
        'active_ctas_per_sm': packed >> 50,
    }


_v272_base_prepare = prepare
_v272_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v272_base_prepare(weights)
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    state['_v272_pair_group'] = {}
    for li, (layer, (n, k, nth)) in enumerate(_V272_SHAPES.items()):
        try:
            if layer not in state.get('cg', {}):
                continue
            pair, pack_exact = _v106_pair_pack(state['cg'][layer])
            entry = {'packed': pair, 'nth': nth, 'enabled': False,
                     'group': 0, 'pack_exact': pack_exact}
            state['_v272_pair_group'][layer] = entry
            gen = torch.Generator(device=wq.device).manual_seed(272101 + li)
            x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                            device=wq.device).to(torch.bfloat16)
            x[0, 0] = 0
            x[0, k - 1] = 16
            data = {'x': x, 'config': {
                'm': 1, 'n': n, 'k': k, 'layer': layer}}
            incumbent = _v272_base_kernel_fn(data, state)
            torch.cuda.synchronize(wq.device)
            best = None
            for group in _V272_GROUPS:
                resources = _v272_resource_info(k, nth, group)
                candidate = _v272_run(x, layer, n, state, group)
                torch.cuda.synchronize(wq.device)
                bitwise = bool(torch.equal(candidate, incumbent))
                w = weights[layer][:n]
                ref = (x.double() @ w.double().t()).to(
                    torch.bfloat16).double()
                rel = float(((candidate.double() - ref).norm() /
                             ref.norm()).item())
                finite = bool(torch.isfinite(candidate).all())
                probe_exact = True
                one = torch.zeros((1, k), dtype=torch.bfloat16,
                                  device=wq.device)
                for col in (0, k // 3, (2 * k) // 3, k - 1):
                    one.zero_()
                    one[0, col] = 1
                    got = _v272_run(one, layer, n, state, group)
                    probe_exact = (probe_exact and
                                   bool(torch.equal(got[0], w[:, col])))
                torch.cuda.synchronize(wq.device)
                cand_ms, base_ms = _v102_paired_graph_ms(
                    lambda g=group: _v272_run(x, layer, n, state, g),
                    lambda: _v272_base_kernel_fn(data, state),
                    wq.device, repeats=13)
                cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
                cand_core = max(cand_us - _V272_REPLAY_OVERHEAD_US, 1e-6)
                base_core = max(base_us - _V272_REPLAY_OVERHEAD_US, 1e-6)
                core_ratio = cand_core / base_core
                valid = (pack_exact and bitwise and probe_exact and finite and
                         rel < 5e-3 and resources['local_bytes'] == 0 and
                         resources['active_ctas_per_sm'] >= 1)
                qualified = (valid and cand_us < base_us and
                             core_ratio <= _V272_REQUIRED_CORE_RATIO)
                print('v272 pair group', layer, 'group', group,
                      'resources', resources, 'pack_exact', pack_exact,
                      'bitwise_base', bitwise, 'probe_exact', probe_exact,
                      'finite', finite, 'fp64_rel', rel,
                      'candidate_us', cand_us, 'incumbent_us', base_us,
                      'core_ratio', core_ratio, 'qualified', qualified,
                      flush=True)
                if qualified and (best is None or cand_us < best[0]):
                    best = (cand_us, group, core_ratio)
            if best is not None:
                entry.update({'enabled': True, 'group': best[1],
                              'core_ratio': best[2]})
                print('v272 pair group selected', layer, 'group', best[1],
                      'candidate_us', best[0], flush=True)
        except Exception as exc:
            print('v272 pair group unavailable', layer, type(exc).__name__,
                  str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    entry = state.get('_v272_pair_group', {}).get(cfg['layer'])
    expected = _V272_SHAPES.get(cfg['layer'])
    if (entry is not None and entry.get('enabled') and cfg['m'] == 1 and
            expected is not None and (cfg['n'], cfg['k']) == expected[:2]):
        x = data['x']
        packed = entry['packed']
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == packed.device and x.data_ptr() % 16 == 0):
            return _v272_run(
                x, cfg['layer'], cfg['n'], state, entry['group'])
    return _v272_base_kernel_fn(data, state)
'''

text += block
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
