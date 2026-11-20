"""Generate v247: isolate the original 8-warp TMA Unpooler/m256 launch.

The exact launch won one prior official paired comparison by 1.21% adjusted
core time and lost another, so it is retested against complete v221 with a 1%
gate.  This is the original rq_deep_tma_pooler arithmetic, not v127's losing
output-store rewrite.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
OUTPUT = ROOT / "submission_plain_unpool256_tma8_margin_v247.py"

text = BASE.read_text(encoding="utf-8").rstrip()
wrapper = r'''

# v247: original v124 8-warp TMA launch, exact m=256 Unpooler only.
_v247_base_prepare = prepare
_v247_base_kernel_fn = kernel_fn
_V247_CFG = (128, 128, 64, 6, 8)


def _v247_run(x, state):
    m, n, k = 256, 8192, 4096
    bm, bn, bk, stages, warps = _V247_CFG
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    wt = state['rq']['t']['unpooler']
    dx = TensorDescriptor.from_tensor(x, [bm, bk])
    dw = TensorDescriptor.from_tensor(wt, [bk, bn])
    rq_deep_tma_pooler[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dx, dw, out, m, n, k, bm, bn, bk,
        EVEN=True, num_warps=warps, num_stages=stages)
    return out


def prepare(weights):
    state = _v247_base_prepare(weights)
    state['_v247_unpool256'] = False
    w = weights.get('unpooler')
    if (w is None or not w.is_cuda or tuple(w.shape) != (8192, 4096) or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device) or
            'rq' not in state):
        return state
    try:
        gen = torch.Generator(device=w.device).manual_seed(247256)
        x = torch.randn((256, 4096), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        probe_cols = (0, 683, 1365, 2048, 3413, 4095)
        for row, col in enumerate(probe_cols):
            x[row].zero_(); x[row, col] = 1
        data = {'x': x, 'config': {
            'layer': 'unpooler', 'm': 256, 'n': 8192, 'k': 4096}}
        candidate = _v247_run(x, state)
        incumbent = _v247_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        finite = bool(torch.isfinite(candidate).all())
        probe_exact = all(torch.equal(candidate[row], w[:, col])
                          for row, col in enumerate(probe_cols))
        rows = torch.arange(11, 8192, 103, device=w.device)[:64]
        ref = x.double() @ w[rows].double().t()
        diff = candidate[:, rows].double() - ref
        rel = float(diff.norm() / ref.norm().clamp_min(1e-30))
        den = torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        worst = float((diff.norm(dim=1) / den).max())
        cand_ms, base_ms = _v102_paired_graph_ms(
            lambda: _v247_run(x, state),
            lambda: _v247_base_kernel_fn(data, state),
            w.device, repeats=13)
        cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
        ratio = max(cand_us - 4.0, 1e-6) / max(base_us - 4.0, 1e-6)
        valid = (finite and probe_exact and rel < 5e-3 and worst < 2e-2 and
                 cand_us < base_us and ratio <= 0.99)
        state['_v247_unpool256'] = valid
        print('v247 Unpool256 TMA8', 'candidate_us', cand_us,
              'incumbent_us', base_us, 'core_ratio', ratio,
              'finite', finite, 'probe_exact', probe_exact,
              'fp64', rel, worst, 'enabled', valid, flush=True)
    except Exception as exc:
        state['_v247_unpool256'] = False
        print('v247 Unpool256 TMA8 unavailable', type(exc).__name__,
              str(exc)[:300], flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    if (state.get('_v247_unpool256') and
            (cfg['layer'], cfg['m'], cfg['n'], cfg['k']) ==
            ('unpooler', 256, 8192, 4096)):
        return _v247_run(data['x'], state)
    return _v247_base_kernel_fn(data, state)
'''

text += "\n\n\n" + wrapper.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
