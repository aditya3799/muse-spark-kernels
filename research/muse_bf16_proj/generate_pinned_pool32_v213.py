"""Generate v213: v211 plus a third independent pooler/m32 guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v211.py"
OUTPUT = ROOT / "submission_plain_pinned_pool32_v213.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v213: a third fresh-data guard for the promoted pooler/m32 route.  Passing
# preserves exactly the v210 timed dispatch; failure removes only this route.
_v213_base_prepare = prepare


def prepare(weights):
    state = _v213_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    if routes.get(('pooler', 32)) != 'slot':
        return state
    w = weights.get('pooler')
    try:
        m, n, k = 32, 4096, 8192
        gen = torch.Generator(device=w.device).manual_seed(213032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        out = _v126_run(x, 'pooler', m, n, k, state)
        torch.cuda.synchronize(w.device)
        finite = bool(torch.isfinite(out).all())
        rows = torch.arange(31, n, 61, device=w.device)[:64]
        ref = x.double() @ w[rows].double().t()
        diff = out[:, rows].double() - ref
        rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
        den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        worst = float((diff.norm(dim=1) / den).max())
        probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
        cols = (2, k // 11, (5 * k) // 11, (9 * k) // 11, k - 3)
        for row, col in enumerate(cols):
            probe[row, col] = 1
        got = _v126_run(probe, 'pooler', m, n, k, state)
        probe_exact = all(torch.equal(got[row], w[:, col])
                          for row, col in enumerate(cols))
        valid = finite and probe_exact and rel < 0.005 and worst < 0.02
        if not valid:
            routes.pop(('pooler', 32), None)
            state['_v134_composite'] = routes
        print('v213 third guard pooler 32 finite', finite,
              'probe_exact', probe_exact, 'fp64', rel, worst,
              'enabled', valid, flush=True)
    except Exception as exc:
        routes.pop(('pooler', 32), None)
        state['_v134_composite'] = routes
        print('v213 third guard unavailable', type(exc).__name__,
              str(exc)[:300], flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
