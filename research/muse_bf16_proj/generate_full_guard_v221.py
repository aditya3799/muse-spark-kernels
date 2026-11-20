"""Generate v221: v210 with an independent guard for every pinned micro-route."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
OUTPUT = ROOT / "submission_plain_full_guard_v221.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v221: independently validate all seven promoted micro-routes on fresh
# activations.  The already validated v197 QKV32 plan retains its own exact
# prepare-time validator.  Passing this guard preserves v210 timed dispatch.
_V221_ROUTES = (
    ('qkv_gate', 8, 2304, 8192, 'slot'),
    ('qkv_gate', 16, 2304, 8192, 'slot'),
    ('o', 16, 8192, 1024, 'slot'),
    ('o', 32, 8192, 1024, 'slot'),
    ('pooler', 1, 4096, 8192, 'pool'),
    ('pooler', 32, 4096, 8192, 'slot'),
    ('unpooler', 32, 8192, 4096, 'l2'),
)
_v221_base_prepare = prepare


def _v221_run(route, x, layer, m, n, k, state):
    if route == 'slot':
        return _v126_run(x, layer, m, n, k, state)
    if route == 'l2':
        return _v134_run_l2(x, state)
    return _v134_run_pool(x, state)


def prepare(weights):
    state = _v221_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    for index, (layer, m, n, k, route) in enumerate(_V221_ROUTES):
        key = (layer, m)
        if routes.get(key) != route:
            print('v221 full guard route absent', key, route, flush=True)
            continue
        try:
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(221000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            out = _v221_run(route, x, layer, m, n, k, state)
            torch.cuda.synchronize(w.device)
            finite = bool(torch.isfinite(out).all())
            rows = torch.arange(13 + index, n, 97, device=w.device)[:32]
            ref = x.double() @ w[rows].double().t()
            diff = out[:, rows].double() - ref
            rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
            den = torch.maximum(ref.norm(dim=1),
                                0.1 * ref.norm(dim=1).mean())
            worst = float((diff.norm(dim=1) / den).max())
            cols = (0, k // 7, k // 3, (2 * k) // 3, k - 1)
            probe_exact = True
            if m == 1:
                for col in cols:
                    probe = torch.zeros((1, k), dtype=torch.bfloat16,
                                        device=w.device)
                    probe[0, col] = 1
                    got = _v221_run(route, probe, layer, m, n, k, state)
                    probe_exact = probe_exact and bool(
                        torch.equal(got[0], w[:n, col]))
            else:
                probe = torch.zeros((m, k), dtype=torch.bfloat16,
                                    device=w.device)
                for row, col in enumerate(cols):
                    probe[row, col] = 1
                got = _v221_run(route, probe, layer, m, n, k, state)
                probe_exact = all(torch.equal(got[row], w[:n, col])
                                  for row, col in enumerate(cols))
            valid = finite and probe_exact and rel < 0.005 and worst < 0.02
            if not valid:
                routes.pop(key, None)
            print('v221 full guard', layer, m, route, 'finite', finite,
                  'probe_exact', probe_exact, 'fp64', rel, worst,
                  'enabled', valid, flush=True)
        except Exception as exc:
            routes.pop(key, None)
            print('v221 full guard unavailable', layer, m,
                  type(exc).__name__, str(exc)[:300], flush=True)
    state['_v134_composite'] = routes
    print('v221 full guard enabled', sorted(routes), flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
