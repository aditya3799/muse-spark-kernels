"""Generate v208: safe follow-up O/pooler m32 tournament on compact v205."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_o32_v205.py"
OUTPUT = ROOT / "submission_plain_sk2_m32_v208.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# -------------------------- v206: O/pooler m32 geometry neighborhood.
_V206_CONFIGS = (
    ('v206_o32_d2_r56', 'o', 8192, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_o32_d4_r112', 'o', 8192, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_o32_a2_r100', 'o', 8192, 32, 2, 2, 2, 2, 2, 1, 32),
    ('v206_o32_w1u4_r100', 'o', 8192, 32, 2, 2, 1, 4, 1, 1, 32),
    ('v206_p32_d4_r112', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_p32_d5_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_p32_d6_r168', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_p32_d7_r196', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1, 32),
    ('v206_p32_a2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 2, 1, 32),
    ('v206_p32_w1u4_r144', 'pooler', 4096, 32, 2, 2, 1, 4, 1, 1, 32),
)
_V206_TOKR = {cfg[0]: cfg[-1] for cfg in _V206_CONFIGS}
SK2_CONFIGS.extend(cfg[:-1] for cfg in _V206_CONFIGS)
_v206_original_sk2_src = _sk2_src


def _sk2_src(sm90):
    if not sm90:
        return _v206_original_sk2_src(sm90)
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            selected = SK2_PICK.get((layer, tokr)) == name
            selected = selected or _V206_TOKR.get(name) == tokr
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


_V206_INDEX = {cfg[0]: i for i, cfg in enumerate(SK2_CONFIGS)}
_v206_base_prepare = prepare
_v206_base_kernel_fn = kernel_fn


def _v206_run(name, x, layer, n, k, state):
    ci, stream, tab, pmax, b4, _ = state['sk2'][name]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16,
                      device=x.device)
    _get_ext().sk2s(stream, x, out, tab, pmax, b4, x.shape[0], n, k,
                    ci, 32, 0, state['dummy'])
    return out


def _v206_validate(name, x, w, layer, n, k, state):
    out = _v206_run(name, x, layer, n, k, state)
    torch.cuda.synchronize(w.device)
    finite = bool(torch.isfinite(out).all())
    rows = torch.linspace(0, n - 1, steps=min(48, n),
                          device=w.device).long()
    ref = x.double() @ w[rows].double().t()
    diff = out[:, rows].double() - ref
    rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    probe = torch.zeros((32, k), dtype=torch.bfloat16, device=w.device)
    cols = (0, k // 3, (2 * k) // 3, k - 1)
    for row, col in enumerate(cols):
        probe[row, col] = 1
    got = _v206_run(name, probe, layer, n, k, state)
    probe_exact = all(torch.equal(got[row], w[:n, col])
                      for row, col in enumerate(cols))
    return out, finite, probe_exact, rel, worst


def prepare(weights):
    state = _v206_base_prepare(weights)
    state['v206_sk2_routes'] = {}
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    try:
        flush = torch.empty((16384, 1024), dtype=torch.float32,
                            device=wq.device)
        for ri, (layer, n, k) in enumerate((('o', 8192, 1024),
                                             ('pooler', 4096, 8192))):
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(206320 + ri)
            x = torch.randn((32, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': 32, 'n': n, 'k': k}}
            incumbent = lambda data=data: _v206_base_kernel_fn(data, state)
            reference = incumbent()
            screened = []
            names = [cfg[0] for cfg in _V206_CONFIGS if cfg[1] == layer]
            for name in names:
                out, finite, probe_exact, rel, worst = _v206_validate(
                    name, x, w, layer, n, k, state)
                bitwise = bool(torch.equal(out, reference))
                valid = finite and probe_exact and rel < 0.005 and worst < 0.02
                samples = []
                if valid:
                    fn = lambda name=name: _v206_run(
                        name, x, layer, n, k, state)
                    samples = [_v150_time(fn, flush, repeats=11)
                               for _ in range(3)]
                    samples.sort()
                    screened.append((samples[1], name))
                print('v206 config', layer, name, 'depth',
                      _sk2_geom(SK2_CONFIGS[_V206_INDEX[name]], 32)[1],
                      'bitwise', bitwise, 'finite', finite,
                      'probe_exact', probe_exact, 'fp64', rel, worst,
                      'samples', samples, flush=True)
            screened.sort()
            print('v206 screen', layer, screened, flush=True)
            paired = []
            for _, name in screened[:3]:
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda name=name: _v206_run(
                        name, x, layer, n, k, state),
                    incumbent, flush, repeats=17, rounds=5)
                paired.append((ratio, candidate_us, incumbent_us, name))
                print('v206 pair', layer, name, 'candidate', candidate_us,
                      'incumbent', incumbent_us, 'core_ratio', ratio,
                      flush=True)
            if paired:
                ratio, candidate_us, incumbent_us, name = min(paired)
                keep = candidate_us < incumbent_us and ratio <= 0.99
                if keep:
                    state['v206_sk2_routes'][(layer, 32)] = name
                print('v206 decision', layer, name, 'core_ratio', ratio,
                      'qualified', keep, flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state['v206_sk2_routes'] = {}
        print('v206 unavailable', type(exc).__name__, str(exc)[:500],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    name = state.get('v206_sk2_routes', {}).get(
        (cfg['layer'], cfg['m']))
    if name is not None:
        return _v206_run(name, data['x'], cfg['layer'], cfg['n'],
                         cfg['k'], state)
    return _v206_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
