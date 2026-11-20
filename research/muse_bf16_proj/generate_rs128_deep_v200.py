"""Generate v200: guarded RS128 stage-6/stage-7 tournament on v198."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pinned_micro_v198.py"
OUTPUT = ROOT / "submission_plain_rs128_deep_v200.py"

text = BASE.read_text(encoding="utf-8")

old_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
  }
"""
new_switch = """  switch (v) {
    case 0: return rq_go<128, 5, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 1: return rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>(p, x, out, (int)KS, (int)info);
    case 2: return rq_go<32, 5, 1, 0, 0, 0, 9437219, 1>(p, x, out, (int)KS, (int)info);
    case 3: return rq_go<128, 6, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
    case 4: return rq_go<128, 7, 1, 0, 0, 0, 42999939, 1>(p, x, out, (int)KS, (int)info);
  }
"""
if text.count(old_switch) != 1:
    raise RuntimeError("unexpected v198 rq switch anchor count")
text = text.replace(old_switch, new_switch, 1)

block = r'''

# ---------------------------------------------- v200: deeper RS128 pipelines.
_V200_TARGETS = ('qkv_gate', 'pooler', 'unpooler')
_V200_VARIANTS = (3, 4)
_v200_base_prepare = prepare
_v200_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v200_base_prepare(weights)
    state['v200_rs128_plans'] = {}
    vstate = state.get('v150')
    if vstate is None:
        return state
    dev = weights['pooler'].device
    try:
        flush = torch.empty((16384, 1024), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(200128)
        winners = {}
        for layer in _V200_TARGETS:
            source = vstate.get('plans', {}).get(layer)
            if source is None:
                continue
            weight = weights[layer]
            x = torch.randn((128, weight.shape[1]), generator=gen,
                            device=dev).to(torch.bfloat16)
            cfg = {'layer': layer, 'm': 128, 'n': weight.shape[0],
                   'k': weight.shape[1]}
            incumbent = lambda x=x, cfg=cfg: _v200_base_kernel_fn(
                {'x': x, 'config': cfg}, state)
            screened = []
            plans = {}
            outputs = {}
            for variant in _V200_VARIANTS:
                try:
                    plan = dict(source)
                    plan['variant'] = variant
                    if not _v158_validate(vstate, plan, weight, layer, 128):
                        continue
                    out = torch.empty((128, weight.shape[0]),
                                      dtype=torch.bfloat16, device=dev)
                    fn = lambda plan=plan, out=out: _v158_call(
                        vstate, plan, x, out)
                    samples = [_v150_time(fn, flush, repeats=9)
                               for _ in range(3)]
                    samples.sort()
                    screened.append((samples[1], variant))
                    plans[variant] = plan
                    outputs[variant] = out
                except Exception as exc:
                    print('v200 variant rejected', layer, variant,
                          type(exc).__name__, str(exc)[:240], flush=True)
            screened.sort()
            print('v200 rs128 screen', layer, screened, flush=True)
            paired = []
            for _, variant in screened:
                plan, out = plans[variant], outputs[variant]
                ratio, candidate_us, incumbent_us = _v150_pair(
                    lambda plan=plan, out=out: _v158_call(
                        vstate, plan, x, out),
                    incumbent, flush, repeats=15, rounds=5)
                paired.append((ratio, candidate_us, incumbent_us, variant))
                print('v200 rs128 pair', layer, variant, 'candidate',
                      candidate_us, 'incumbent', incumbent_us,
                      'core_ratio', ratio, flush=True)
            if paired:
                ratio, candidate_us, incumbent_us, variant = min(paired)
                keep = candidate_us < incumbent_us and ratio <= 0.995
                if keep:
                    winners[layer] = plans[variant]
                print('v200 rs128 decision', layer, variant,
                      'core_ratio', ratio, 'qualified', keep, flush=True)
        state['v200_rs128_plans'] = winners
        print('v200 rs128 enabled', sorted(winners), flush=True)
        del flush
        torch.cuda.empty_cache()
    except Exception as exc:
        state['v200_rs128_plans'] = {}
        print('v200 unavailable', type(exc).__name__, str(exc)[:400],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    plan = state.get('v200_rs128_plans', {}).get(cfg['layer'])
    if cfg['m'] == 128 and plan is not None:
        return _v158_call(state['v150'], plan, data['x'])
    return _v200_base_kernel_fn(data, state)
'''

text = text.rstrip() + "\n" + block.lstrip("\n")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
