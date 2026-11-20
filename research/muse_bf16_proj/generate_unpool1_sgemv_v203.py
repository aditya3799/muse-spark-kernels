"""Generate v203: safe uncompressed unpooler/m1 GEMV geometry tournament."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_trimmed_qkv32_v197.py"
OUTPUT = ROOT / "submission_plain_unpool1_sgemv_v203.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# ----------------------------- v203: unpooler/m1 BF16 GEMV geometry sweep.
# Unlike compressed cgemv, this kernel has no row-grouped escape table, so its
# rows-per-CTA geometry can be changed without reinterpreting auxiliary data.
CPP_SRC += r"""
void sgemv_tune(torch::Tensor W, torch::Tensor x, torch::Tensor out,
                 int64_t N, int64_t rows, int64_t nth);
"""

RGEMV_SRC += r"""
void sgemv_tune(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N,
                 int64_t rows, int64_t nth) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % rows == 0 &&
              N <= W.size(0), "sgemv_tune: m = 1, K = 4096");
  auto wp = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto st = at::cuda::getCurrentCUDAStream();
  if (rows == 1 && nth == 512)
    sgemv_kernel<1, 512, 4096><<<N, 512, 0, st>>>(wp, xp, op);
  else if (rows == 2 && nth == 512)
    sgemv_kernel<2, 512, 4096><<<N / 2, 512, 0, st>>>(wp, xp, op);
  else if (rows == 8 && nth == 512)
    sgemv_kernel<8, 512, 4096><<<N / 8, 512, 0, st>>>(wp, xp, op);
  else if (rows == 16 && nth == 512)
    sgemv_kernel<16, 512, 4096><<<N / 16, 512, 0, st>>>(wp, xp, op);
  else if (rows == 2 && nth == 256)
    sgemv_kernel<2, 256, 4096><<<N / 2, 256, 0, st>>>(wp, xp, op);
  else if (rows == 4 && nth == 256)
    sgemv_kernel<4, 256, 4096><<<N / 4, 256, 0, st>>>(wp, xp, op);
  else if (rows == 8 && nth == 256)
    sgemv_kernel<8, 256, 4096><<<N / 8, 256, 0, st>>>(wp, xp, op);
  else
    TORCH_CHECK(false, "sgemv_tune: unsupported rows/threads");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

# The active extension loader has a fixed function list. Intercept its single
# load_inline call and add the readable sgemv_tune declaration to that list.
_v203_load_inline = load_inline


def load_inline(*args, **kwargs):
    functions = kwargs.get('functions')
    if functions is not None and 'sgemv' in functions and 'sgemv_tune' not in functions:
        kwargs['functions'] = list(functions) + ['sgemv_tune']
    return _v203_load_inline(*args, **kwargs)


_V203_CONFIGS = ((1, 512), (2, 512), (8, 512), (16, 512),
                 (2, 256), (4, 256), (8, 256))
_v203_base_prepare = prepare
_v203_base_kernel_fn = kernel_fn


def _v203_run(x, state, rows, nth):
    out = torch.empty((1, 1024), dtype=torch.bfloat16, device=x.device)
    _get_ext().sgemv_tune(state['raw']['unpooler'], x, out, 1024,
                          rows, nth)
    return out


def prepare(weights):
    state = _v203_base_prepare(weights)
    state['v203_unpool1_config'] = None
    w = weights.get('unpooler')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        n, k = 1024, 4096
        gen = torch.Generator(device=w.device).manual_seed(203001)
        x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        cfg = {'layer': 'unpooler', 'm': 1, 'n': n, 'k': k}
        incumbent = lambda: _v203_base_kernel_fn(
            {'x': x, 'config': cfg}, state)
        reference = incumbent()
        flush = torch.empty((16384, 1024), dtype=torch.float32,
                            device=w.device)
        screened = []
        for rows, nth in _V203_CONFIGS:
            candidate = _v203_run(x, state, rows, nth)
            torch.cuda.synchronize(w.device)
            exact = bool(torch.equal(candidate, reference))
            finite = bool(torch.isfinite(candidate).all())
            probe_exact = True
            one = torch.zeros((1, k), dtype=torch.bfloat16, device=w.device)
            for col in (0, k // 3, (2 * k) // 3, k - 1):
                one.zero_()
                one[0, col] = 1
                got = _v203_run(one, state, rows, nth)
                probe_exact = probe_exact and bool(
                    torch.equal(got[0], w[:n, col]))
            samples = []
            if exact and finite and probe_exact:
                fn = lambda rows=rows, nth=nth: _v203_run(
                    x, state, rows, nth)
                samples = [_v150_time(fn, flush, repeats=11)
                           for _ in range(3)]
                samples.sort()
                screened.append((samples[1], rows, nth))
            print('v203 unpool1 config', rows, nth, 'bitwise', exact,
                  'finite', finite, 'probe_exact', probe_exact,
                  'samples', samples, flush=True)
        screened.sort()
        print('v203 unpool1 screen', screened, flush=True)
        paired = []
        for _, rows, nth in screened[:3]:
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda rows=rows, nth=nth: _v203_run(x, state, rows, nth),
                incumbent, flush, repeats=17, rounds=5)
            paired.append((ratio, candidate_us, incumbent_us, rows, nth))
            print('v203 unpool1 pair', rows, nth,
                  'candidate', candidate_us, 'incumbent', incumbent_us,
                  'core_ratio', ratio, flush=True)
        if paired:
            ratio, candidate_us, incumbent_us, rows, nth = min(paired)
            keep = candidate_us < incumbent_us and ratio <= 0.995
            if keep:
                state['v203_unpool1_config'] = (rows, nth)
            print('v203 unpool1 decision', rows, nth,
                  'core_ratio', ratio, 'qualified', keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state['v203_unpool1_config'] = None
        print('v203 unavailable', type(exc).__name__, str(exc)[:400],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    choice = state.get('v203_unpool1_config')
    if cfg['m'] == 1 and cfg['layer'] == 'unpooler' and choice is not None:
        return _v203_run(data['x'], state, *choice)
    return _v203_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
