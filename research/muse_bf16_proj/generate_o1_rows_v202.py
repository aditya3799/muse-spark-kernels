"""Generate v202: O/m1 C12 GEMV row-group/thread tournament on v197."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_trimmed_qkv32_v197.py"
OUTPUT = ROOT / "submission_plain_o1_rows_v202.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# ----------------------------------------- v202: O/m1 row/thread tournament.
_v202_dispatch_anchor = r"""else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);"""
_v202_dispatch_replacement = r"""else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 1 && nth == 64) cgemv_kernel<1, 64, 1024><<<N, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 4 && nth == 64) cgemv_kernel<4, 64, 1024><<<N / 4, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 1 && nth == 128) cgemv_kernel<1, 128, 1024><<<N, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 1024><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 4 && nth == 128) cgemv_kernel<4, 128, 1024><<<N / 4, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);"""
if RGEMV_SRC.count(_v202_dispatch_anchor) != 1:
    raise RuntimeError('v202 cgemv dispatch anchor mismatch')
RGEMV_SRC = RGEMV_SRC.replace(
    _v202_dispatch_anchor, _v202_dispatch_replacement, 1)

_v202_info_anchor = r"""    case 5: return cgemv_info_t<2, 256, 4096>();
    default: TORCH_CHECK(false, "cgemv_info: bad instance");"""
_v202_info_replacement = r"""    case 5: return cgemv_info_t<2, 256, 4096>();
    case 6: return cgemv_info_t<1, 64, 1024>();
    case 7: return cgemv_info_t<4, 64, 1024>();
    case 8: return cgemv_info_t<1, 128, 1024>();
    case 9: return cgemv_info_t<2, 128, 1024>();
    case 10: return cgemv_info_t<4, 128, 1024>();
    default: TORCH_CHECK(false, "cgemv_info: bad instance");"""
if RGEMV_SRC.count(_v202_info_anchor) != 1:
    raise RuntimeError('v202 cgemv info anchor mismatch')
RGEMV_SRC = RGEMV_SRC.replace(_v202_info_anchor, _v202_info_replacement, 1)

_V202_CONFIGS = ((1, 64, 6), (4, 64, 7), (1, 128, 8),
                 (2, 128, 9), (4, 128, 10))
_v202_base_prepare = prepare
_v202_base_kernel_fn = kernel_fn


def _v202_resource(which):
    packed = int(_get_ext().cgemv_info(which))
    return {
        'registers': packed & ((1 << 10) - 1),
        'local_bytes': (packed >> 10) & ((1 << 20) - 1),
        'static_shared_bytes': (packed >> 30) & ((1 << 20) - 1),
        'active_ctas_per_sm': packed >> 50,
    }


def prepare(weights):
    state = _v202_base_prepare(weights)
    state['v202_o1_config'] = None
    w = weights.get('o')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        n, k = 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(202001)
        x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        cfg = {'layer': 'o', 'm': 1, 'n': n, 'k': k}
        incumbent = lambda: _v202_base_kernel_fn(
            {'x': x, 'config': cfg}, state)
        flush = torch.empty((16384, 1024), dtype=torch.float32,
                            device=w.device)
        screened = []
        for rows, nth, which in _V202_CONFIGS:
            candidate = _v105_run_cgemv(x, 'o', n, state, rows, nth)
            reference = incumbent()
            torch.cuda.synchronize(w.device)
            exact = bool(torch.equal(candidate, reference))
            finite = bool(torch.isfinite(candidate).all())
            probe_exact = True
            one = torch.zeros((1, k), dtype=torch.bfloat16, device=w.device)
            for col in (0, k // 3, (2 * k) // 3, k - 1):
                one.zero_()
                one[0, col] = 1
                got = _v105_run_cgemv(one, 'o', n, state, rows, nth)
                probe_exact = probe_exact and bool(torch.equal(got[0], w[:, col]))
            resources = _v202_resource(which)
            valid = (exact and finite and probe_exact and
                     resources['local_bytes'] == 0 and
                     resources['active_ctas_per_sm'] >= 1)
            samples = []
            if valid:
                fn = lambda rows=rows, nth=nth: _v105_run_cgemv(
                    x, 'o', n, state, rows, nth)
                samples = [_v150_time(fn, flush, repeats=11)
                           for _ in range(3)]
                samples.sort()
                screened.append((samples[1], rows, nth))
            print('v202 o1 config', rows, nth, 'resources', resources,
                  'bitwise', exact, 'finite', finite,
                  'probe_exact', probe_exact, 'samples', samples, flush=True)
        screened.sort()
        print('v202 o1 screen', screened, flush=True)
        paired = []
        for _, rows, nth in screened[:3]:
            ratio, candidate_us, incumbent_us = _v150_pair(
                lambda rows=rows, nth=nth: _v105_run_cgemv(
                    x, 'o', n, state, rows, nth),
                incumbent, flush, repeats=17, rounds=5)
            paired.append((ratio, candidate_us, incumbent_us, rows, nth))
            print('v202 o1 pair', rows, nth, 'candidate', candidate_us,
                  'incumbent', incumbent_us, 'core_ratio', ratio, flush=True)
        if paired:
            ratio, candidate_us, incumbent_us, rows, nth = min(paired)
            keep = candidate_us < incumbent_us and ratio <= 0.995
            if keep:
                state['v202_o1_config'] = (rows, nth)
            print('v202 o1 decision', rows, nth, 'core_ratio', ratio,
                  'qualified', keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state['v202_o1_config'] = None
        print('v202 unavailable', type(exc).__name__, str(exc)[:400],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    choice = state.get('v202_o1_config')
    if cfg['m'] == 1 and cfg['layer'] == 'o' and choice is not None:
        rows, nth = choice
        return _v105_run_cgemv(data['x'], 'o', cfg['n'], state, rows, nth)
    return _v202_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
