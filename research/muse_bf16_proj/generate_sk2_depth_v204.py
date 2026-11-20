"""Generate v204: pipeline-depth/accumulator sweep around proven SK2 routes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_micro_v198.py"
OUTPUT = ROOT / "submission_plain_sk2_depth_v204.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# ---------------- v204: depth/accumulator neighborhood of proven SK2 routes.
# Each tuple is ordinary readable CUDA-template configuration data.  Ring caps
# select pipeline depth through the existing _sk2_geom calculation.
_V204_CONFIGS = (
    ('v204_q8_d4_r112', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1, 8),
    ('v204_q8_d6_r168', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1, 8),
    ('v204_q8_a1_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1, 8),
    ('v204_q16_d4_r148', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1, 16),
    ('v204_q16_d6_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1, 16),
    ('v204_q16_a1_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1, 16),
    ('v204_o16_d2_r48', 'o', 8192, 32, 2, 2, 1, 4, 2, 1, 16),
    ('v204_o16_d4_r88', 'o', 8192, 32, 2, 2, 1, 4, 2, 1, 16),
    ('v204_o16_d5_r108', 'o', 8192, 32, 2, 2, 1, 4, 2, 1, 16),
    ('v204_o16_a1_r68', 'o', 8192, 32, 2, 2, 1, 4, 1, 1, 16),
)
_V204_TOKR = {cfg[0]: cfg[-1] for cfg in _V204_CONFIGS}
SK2_CONFIGS.extend(cfg[:-1] for cfg in _V204_CONFIGS)

# Compile the production picks plus the explicitly enumerated tournament
# instances.  This is the existing source generator with one additional,
# narrowly-scoped eligibility condition.
_v204_original_sk2_src = _sk2_src


def _sk2_src(sm90):
    if not sm90:
        return _v204_original_sk2_src(sm90)
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            selected = SK2_PICK.get((layer, tokr)) == name
            selected = selected or _V204_TOKR.get(name) == tokr
            if not selected:
                continue
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = 'float' if hl else '__nv_bfloat16'
            nacc = 1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC
            lines.append(
                f'if (cfg == {ci} && tokr == {tokr}) {{ '
                f'launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, '
                f'{nstage}, {str(not hl).lower()}, {str(hl).lower()}, '
                f'{nacc}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, '
                f'M, N, K, dg, dbg); return; }}')
    return SK2_SRC.replace('@@SK2_DISPATCH@@', '\n  '.join(lines))


_V204_INDEX = {cfg[0]: i for i, cfg in enumerate(SK2_CONFIGS)}
_v204_base_prepare = prepare
_v204_base_kernel_fn = kernel_fn


def _v204_run(name, x, layer, n, k, state):
    ci, stream, tab, pmax, b4, _ = state['sk2'][name]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16,
                      device=x.device)
    _get_ext().sk2s(stream, x, out, tab, pmax, b4, x.shape[0], n, k,
                    ci, _V204_TOKR[name], 0, state['dummy'])
    return out


def prepare(weights):
    state = _v204_base_prepare(weights)
    state['v204_sk2_routes'] = {}
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    try:
        flush = torch.empty((16384, 1024), dtype=torch.float32,
                            device=wq.device)
        routes = (('qkv_gate', 8, 2304, 8192),
                  ('qkv_gate', 16, 2304, 8192),
                  ('o', 16, 8192, 1024))
        for ri, (layer, m, n, k) in enumerate(routes):
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(204100 + ri)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': m, 'n': n, 'k': k}}
            incumbent = lambda data=data: _v204_base_kernel_fn(data, state)
            reference = incumbent()
            screened = []
            names = [cfg[0] for cfg in _V204_CONFIGS
                     if cfg[1] == layer and cfg[-1] == m]
            for name in names:
                candidate = _v204_run(name, x, layer, n, k, state)
                torch.cuda.synchronize(w.device)
                exact = bool(torch.equal(candidate, reference))
                finite = bool(torch.isfinite(candidate).all())
                probe = torch.zeros((m, k), dtype=torch.bfloat16,
                                    device=w.device)
                cols = (0, k // 3, (2 * k) // 3, k - 1)
                for row, col in enumerate(cols):
                    probe[row, col] = 1
                got = _v204_run(name, probe, layer, n, k, state)
                probe_exact = all(torch.equal(got[row], w[:n, col])
                                  for row, col in enumerate(cols))
                samples = []
                if exact and finite and probe_exact:
                    fn = lambda name=name: _v204_run(
                        name, x, layer, n, k, state)
                    samples = [_v150_time(fn, flush, repeats=11)
                               for _ in range(3)]
                    samples.sort()
                    screened.append((samples[1], name))
                print('v204 config', layer, m, name, 'depth',
                      _sk2_geom(SK2_CONFIGS[_V204_INDEX[name]], m)[1],
                      'bitwise', exact, 'finite', finite,
                      'probe_exact', probe_exact, 'samples', samples,
                      flush=True)
            screened.sort()
            print('v204 screen', layer, m, screened, flush=True)
            paired = []
            for _, name in screened[:2]:
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda name=name: _v204_run(
                        name, x, layer, n, k, state),
                    incumbent, flush, repeats=17, rounds=5)
                paired.append((ratio, candidate_us, incumbent_us, name))
                print('v204 pair', layer, m, name,
                      'candidate', candidate_us, 'incumbent', incumbent_us,
                      'core_ratio', ratio, flush=True)
            if paired:
                ratio, candidate_us, incumbent_us, name = min(paired)
                keep = candidate_us < incumbent_us and ratio <= 0.995
                if keep:
                    state['v204_sk2_routes'][(layer, m)] = name
                print('v204 decision', layer, m, name,
                      'core_ratio', ratio, 'qualified', keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state['v204_sk2_routes'] = {}
        print('v204 unavailable', type(exc).__name__, str(exc)[:500],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    name = state.get('v204_sk2_routes', {}).get(
        (cfg['layer'], cfg['m']))
    if name is not None:
        return _v204_run(name, data['x'], cfg['layer'], cfg['n'],
                         cfg['k'], state)
    return _v204_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
