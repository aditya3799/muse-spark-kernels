"""Generate v205: compact v198 plus deterministic, bitwise-guarded O/m32."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_micro_v198.py"
OUTPUT = ROOT / "submission_plain_pinned_o32_v205.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# --------------------------- v205: repeated exact O/m32 micro-route win.
# The same readable slot-owner kernel was bitwise exact in every recent run and
# won the last two official paired comparisons.  Revalidate it once more on
# this H200, then keep it independent of the noisy sub-microsecond timing gate.
_v205_base_prepare = prepare


def prepare(weights):
    state = _v205_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    w = weights.get('o')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        m, n, k = 32, 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(205032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': m, 'n': n, 'k': k}}
        candidate = _v126_run(x, 'o', m, n, k, state)
        incumbent = _v134_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        exact = bool(torch.equal(candidate, incumbent))
        finite = bool(torch.isfinite(candidate).all())
        probe = torch.zeros((m, k), dtype=torch.bfloat16, device=w.device)
        cols = (0, k // 3, (2 * k) // 3, k - 1)
        for row, col in enumerate(cols):
            probe[row, col] = 1
        got = _v126_run(probe, 'o', m, n, k, state)
        probe_exact = all(torch.equal(got[row], w[:, col])
                          for row, col in enumerate(cols))
        if exact and finite and probe_exact:
            routes[('o', 32)] = 'slot'
        else:
            routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v205 pin o 32 slot bitwise', exact, 'finite', finite,
              'probe_exact', probe_exact,
              'enabled', exact and finite and probe_exact, flush=True)
    except Exception as exc:
        routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v205 pin unavailable', type(exc).__name__, str(exc)[:300],
              flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
