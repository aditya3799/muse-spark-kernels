"""Generate v209: v207 with a third independent O/m32 exactness guard."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_o32_v207.py"
OUTPUT = ROOT / "submission_plain_pinned_o32_v209.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# v209 adds a third independent validation activation for the pinned O/m32 path.
_v209_base_prepare = prepare


def prepare(weights):
    state = _v209_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    if routes.get(('o', 32)) != 'slot':
        return state
    w = weights.get('o')
    try:
        m, n, k = 32, 8192, 1024
        gen = torch.Generator(device=w.device).manual_seed(209032)
        x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': m, 'n': n, 'k': k}}
        candidate = _v126_run(x, 'o', m, n, k, state)
        incumbent = _v134_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        exact = bool(torch.equal(candidate, incumbent))
        finite = bool(torch.isfinite(candidate).all())
        if not (exact and finite):
            routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v209 third guard o 32 bitwise', exact, 'finite', finite,
              'enabled', exact and finite, flush=True)
    except Exception as exc:
        routes.pop(('o', 32), None)
        state['_v134_composite'] = routes
        print('v209 guard unavailable', type(exc).__name__, str(exc)[:240],
              flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
