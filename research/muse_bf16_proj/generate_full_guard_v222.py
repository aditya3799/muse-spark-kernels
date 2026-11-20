"""Generate v222: v221 plus a second independent all-route guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
OUTPUT = ROOT / "submission_plain_full_guard_v222.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v222: repeat the all-route numerical guard with independent activations and
# coefficient columns.  Passing changes no timed dispatch relative to v210.
_v222_base_prepare = prepare


def prepare(weights):
    state = _v222_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    for index, (layer, m, n, k, route) in enumerate(_V221_ROUTES):
        key = (layer, m)
        if routes.get(key) != route:
            continue
        try:
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(222000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            out = _v221_run(route, x, layer, m, n, k, state)
            torch.cuda.synchronize(w.device)
            finite = bool(torch.isfinite(out).all())
            rows = torch.arange(29 + index, n, 89, device=w.device)[:32]
            ref = x.double() @ w[rows].double().t()
            diff = out[:, rows].double() - ref
            rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
            den = torch.maximum(ref.norm(dim=1),
                                0.1 * ref.norm(dim=1).mean())
            worst = float((diff.norm(dim=1) / den).max())
            cols = (1, k // 11, (4 * k) // 11, (8 * k) // 11, k - 2)
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
            print('v222 second full guard', layer, m, route,
                  'finite', finite, 'probe_exact', probe_exact,
                  'fp64', rel, worst, 'enabled', valid, flush=True)
        except Exception as exc:
            routes.pop(key, None)
            print('v222 second full guard unavailable', layer, m,
                  type(exc).__name__, str(exc)[:300], flush=True)
    state['_v134_composite'] = routes
    print('v222 second full guard enabled', sorted(routes), flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
