"""Generate v212: cluster-2 split-K tournament for underfilled small routes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
OUTPUT = ROOT / "submission_plain_cluster2_small_v212.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v212: QKV exposes 96 single-CTA row blocks and pooler exposes 128.  These
# cluster-2 neighbors double flight count and split K in half while preserving
# the exact C12 representation and the existing slot-owner arithmetic.
_V212_CONFIGS = (
    ('v212_q8_cl2_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 8),
    ('v212_q16_cl2_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 2, 16),
    ('v212_q32_cl2_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2, 32),
    ('v212_p16_cl2_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 2, 16),
    ('v212_p32_cl2_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 2, 32),
)
_V212_TOKR = {cfg[0]: cfg[-1] for cfg in _V212_CONFIGS}
SK2_CONFIGS.extend(cfg[:-1] for cfg in _V212_CONFIGS)
_v212_original_sk2_src = _sk2_src


def _sk2_src(sm90):
    if not sm90:
        return _v212_original_sk2_src(sm90)
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            selected = SK2_PICK.get((layer, tokr)) == name
            selected = selected or _V212_TOKR.get(name) == tokr
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


_V212_INDEX = {cfg[0]: i for i, cfg in enumerate(SK2_CONFIGS)}
_v212_base_prepare = prepare
_v212_base_kernel_fn = kernel_fn


def _v212_run(name, x, layer, n, k, state):
    ci, stream, tab, pmax, b4, _ = state['sk2'][name]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16,
                      device=x.device)
    _get_ext().sk2s(stream, x, out, tab, pmax, b4, x.shape[0], n, k,
                    ci, _V212_TOKR[name], 0, state['dummy'])
    return out


def _v212_validate(name, x, w, layer, n, k, state):
    out = _v212_run(name, x, layer, n, k, state)
    torch.cuda.synchronize(w.device)
    finite = bool(torch.isfinite(out).all())
    rows = torch.linspace(0, n - 1, steps=min(64, n),
                          device=w.device).long()
    ref = x.double() @ w[rows].double().t()
    diff = out[:, rows].double() - ref
    rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
    den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
    worst = float((diff.norm(dim=1) / den).max())
    probe = torch.zeros((x.shape[0], k), dtype=torch.bfloat16,
                        device=w.device)
    cols = (0, k // 7, k // 3, (2 * k) // 3, k - 1)
    for row, col in enumerate(cols[:x.shape[0]]):
        probe[row, col] = 1
    got = _v212_run(name, probe, layer, n, k, state)
    probe_exact = all(torch.equal(got[row], w[:n, col])
                      for row, col in enumerate(cols[:x.shape[0]]))
    valid = finite and probe_exact and rel < 0.005 and worst < 0.02
    return valid, finite, probe_exact, rel, worst


def prepare(weights):
    state = _v212_base_prepare(weights)
    state['v212_routes'] = {}
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    try:
        flush = torch.empty((16384, 1024), dtype=torch.float32,
                            device=wq.device)
        for index, cfg in enumerate(_V212_CONFIGS):
            name, layer, n, _, _, _, _, _, _, _, m = cfg
            k = weights[layer].shape[1]
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(212000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': m, 'n': n, 'k': k}}
            valid, finite, probe_exact, rel, worst = _v212_validate(
                name, x, w, layer, n, k, state)
            candidate = lambda name=name, x=x, layer=layer, n=n, k=k: (
                _v212_run(name, x, layer, n, k, state))
            incumbent = lambda data=data: _v212_base_kernel_fn(data, state)
            ratio, candidate_us, incumbent_us = _v150_pair(
                candidate, incumbent, flush, repeats=17, rounds=5)
            keep = (valid and candidate_us < incumbent_us and ratio <= 0.99)
            if keep:
                state['v212_routes'][(layer, m)] = name
            info = _v126_resources(layer, m, state)
            print('v212 cluster2', layer, m, name,
                  'depth', _sk2_geom(SK2_CONFIGS[_V212_INDEX[name]], m)[1],
                  'finite', finite, 'probe_exact', probe_exact,
                  'fp64', rel, worst, 'resources_reference', info,
                  'candidate', candidate_us, 'incumbent', incumbent_us,
                  'core_ratio', ratio, 'qualified', keep, flush=True)
        del flush
        torch.cuda.empty_cache()
        print('v212 enabled', sorted(state['v212_routes']), flush=True)
    except Exception as exc:
        state['v212_routes'] = {}
        print('v212 unavailable', type(exc).__name__, str(exc)[:500],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    name = state.get('v212_routes', {}).get((cfg['layer'], cfg['m']))
    if name is not None:
        return _v212_run(name, data['x'], cfg['layer'], cfg['n'],
                         cfg['k'], state)
    return _v212_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
