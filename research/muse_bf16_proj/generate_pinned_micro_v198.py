"""Generate v198: v197 plus deterministic dispatch for repeated micro-wins.

Five v134 routes won and were bitwise exact in both the v192 check and score
runs.  Recheck their outputs during prepare, then pin them independently of
the noisy sub-microsecond timing gate.  Marginal O/m32 and Pooler/m32 routes
remain governed by v192's original adaptive decision.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_trimmed_qkv32_v197.py"
OUTPUT = ROOT / "submission_plain_pinned_micro_v198.py"

text = BASE.read_text(encoding="utf-8")

block = r'''

# -------------------------------------------- v198: repeated exact micro-wins.
_V198_PINNED = (
    ('qkv_gate', 8, 2304, 8192, 'slot'),
    ('qkv_gate', 16, 2304, 8192, 'slot'),
    ('o', 16, 8192, 1024, 'slot'),
    ('unpooler', 32, 8192, 4096, 'l2'),
    ('pooler', 1, 4096, 8192, 'pool'),
)
_v198_base_prepare = prepare


def prepare(weights):
    state = _v198_base_prepare(weights)
    routes = dict(state.get('_v134_composite', {}))
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    for index, (layer, m, n, k, route) in enumerate(_V198_PINNED):
        try:
            w = weights[layer]
            gen = torch.Generator(device=w.device).manual_seed(198000 + index)
            x = torch.randn((m, k), generator=gen, dtype=torch.float32,
                            device=w.device).to(torch.bfloat16)
            data = {'x': x, 'config': {
                'layer': layer, 'm': m, 'n': n, 'k': k}}
            if route == 'slot':
                candidate = _v126_run(x, layer, m, n, k, state)
            elif route == 'l2':
                candidate = _v134_run_l2(x, state)
            else:
                candidate = _v134_run_pool(x, state)
            incumbent = _v134_base_kernel_fn(data, state)
            torch.cuda.synchronize(w.device)
            exact = bool(torch.equal(candidate, incumbent))
            finite = bool(torch.isfinite(candidate).all())
            if exact and finite:
                routes[(layer, m)] = route
            else:
                routes.pop((layer, m), None)
            print('v198 pin', layer, m, route, 'bitwise', exact,
                  'finite', finite, 'enabled', exact and finite, flush=True)
        except Exception as exc:
            routes.pop((layer, m), None)
            print('v198 pin unavailable', layer, m, type(exc).__name__,
                  str(exc)[:240], flush=True)
    state['_v134_composite'] = routes
    print('v198 pinned composite enabled', sorted(routes), flush=True)
    return state
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
