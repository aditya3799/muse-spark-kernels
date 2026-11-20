"""Generate v211: v210 plus an independent pooler/m32 validation guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_pool32_v210.py"
OUTPUT = ROOT / "submission_plain_pinned_pool32_v211.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v211: independently revalidate the promoted pooler/m32 route on fresh data.
# This guard changes no timed dispatch when the existing exact-coefficient and
# FP64-quality requirements hold.
_v211_base_prepare = prepare


def prepare(weights):
    state = _v211_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    if routes.get(('pooler', 32)) != 'slot':
        return state
    w = weights.get('pooler')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        m, n, k = 32, 4096, 8192
        gen = torch.Generator(device=w.device).manual_seed(211032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        out = _v126_run(x, 'pooler', m, n, k, state)
        torch.cuda.synchronize(w.device)
        finite = bool(torch.isfinite(out).all())
        rows = torch.arange(17, n, 67, device=w.device)[:64]
        ref = x.double() @ w[rows].double().t()
        diff = out[:, rows].double() - ref
        rel = float(diff.norm() / ref.norm().clamp_min(1e-20))
        den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        worst = float((diff.norm(dim=1) / den).max())
        probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
        cols = (1, k // 7, (3 * k) // 7, (6 * k) // 7, k - 2)
        for row, col in enumerate(cols):
            probe[row, col] = 1
        got = _v126_run(probe, 'pooler', m, n, k, state)
        probe_exact = all(torch.equal(got[row], w[:, col])
                          for row, col in enumerate(cols))
        valid = finite and probe_exact and rel < 0.005 and worst < 0.02
        if not valid:
            routes.pop(('pooler', 32), None)
            state['_v134_composite'] = routes
        print('v211 second guard pooler 32 finite', finite,
              'probe_exact', probe_exact, 'fp64', rel, worst,
              'enabled', valid, flush=True)
    except Exception as exc:
        routes.pop(('pooler', 32), None)
        state['_v134_composite'] = routes
        print('v211 second guard unavailable', type(exc).__name__,
              str(exc)[:300], flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
