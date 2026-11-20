"""Generate v243: a genuinely distinct O/m256 Hopper configuration tournament.

v221 already contains v56's (256,64,5,7,1,1,1) kernel exactly, so this
candidate keeps that configuration only as a compile/control point and tests a
bounded neighborhood of stage depth, pipeline policy, K tile, and warp-group
geometry.  A distinct plan dispatches only after exhaustive coefficient probes,
FP64 validation, zero-local-memory validation, and a 1% adjusted-core win over
the complete v221 route.  The output contains literal readable CUDA.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_plain_o256_margin_v230.py"
OUTPUT = ROOT / "submission_plain_o256_config_tournament_v243.py"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


base = BASE.read_text(encoding="utf-8").rstrip()
donor = DONOR.read_text(encoding="utf-8")
start = donor.index("_o256_RL_CUDA = r'''")
end = donor.index("# v230: independently validate", start)
support = donor[start:end].rstrip()

configs = """_o256_CONFIGS = [
    (256, 64, 5, 7, 1, 1, 1),  # v221/v56 control; never promoted here
    (256, 64, 5, 7, 1, 0, 1),  # ordinary L2 policy
    (256, 64, 4, 7, 1, 1, 1),
    (256, 64, 3, 7, 1, 1, 1),
    (256, 64, 5, 6, 1, 1, 1),
    (256, 64, 5, 3, 1, 1, 1),
    (256, 128, 2, 7, 1, 1, 1),
    (256, 128, 1, 6, 1, 1, 1),
    (128, 64, 5, 7, 1, 1, 2),
    (128, 64, 4, 7, 1, 1, 2),
    (128, 128, 2, 7, 1, 1, 2),
    (128, 64, 5, 7, 1, 1, 1),
    (64, 64, 5, 7, 1, 1, 2),
]"""
support = replace_once(
    support,
    "_o256_CONFIGS = [(256, 64, 5, 7, 1, 1, 1)]",
    configs,
)
support = replace_once(
    support,
    "def _o256_launch(x, state, query=0):",
    "def _o256_launch(x, state, query=0, inst=0):",
)
support = support.replace(
    "            0, x.data_ptr(), state['weight'].data_ptr(), 0, 256,",
    "            inst, x.data_ptr(), state['weight'].data_ptr(), 0, 256,",
    1,
)
support = support.replace(
    "        0, x.data_ptr(), state['weight'].data_ptr(), out.data_ptr(), 256,",
    "        inst, x.data_ptr(), state['weight'].data_ptr(), out.data_ptr(), 256,",
    1,
)
support = replace_once(
    support,
    "def _o256_validate(state, weight):",
    "def _o256_validate(state, weight, inst):",
)
support = replace_once(
    support,
    "attr = _o256_launch(probe, state, query=1)",
    "attr = _o256_launch(probe, state, query=1, inst=inst)",
)
support = support.replace(
    "got = _o256_launch(x, state)",
    "got = _o256_launch(x, state, inst=inst)",
)
support = replace_once(
    support,
    "state['valid'] = _o256_validate(state, weight)",
    "state['valid'] = True",
)

wrapper = r'''

# v243: test only distinct configurations.  The byte-identical control at id 0
# documents the incumbent but can never be selected by this wrapper.
_v243_base_prepare = prepare
_v243_base_kernel_fn = kernel_fn


def prepare(weights):
    started = _v150_clock.time()
    build_result = []

    def build():
        try:
            build_result.append(('ok', _o256_ext()))
        except Exception as exc:
            build_result.append(('error', type(exc).__name__, str(exc)[:300]))

    build_thread = _v150_threading.Thread(target=build, daemon=True)
    build_thread.start()
    state = _v243_base_prepare(weights)
    state['_v243_o256'] = {'enabled': False}
    w = weights.get('o')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        build_thread.join(timeout=max(
            1.0, 285.0 - (_v150_clock.time() - started)))
        if not build_result or build_result[0][0] != 'ok':
            print('v243 O256 build unavailable', build_result, flush=True)
            return state
        plan = _o256_prepare(weights)
        state['_v243_o256'] = plan
        if not plan.get('valid'):
            return state
        gen = torch.Generator(device=w.device).manual_seed(243256)
        x = torch.randn((256, 1024), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': 256, 'n': 8192, 'k': 1024}}
        incumbent = lambda: _v243_base_kernel_fn(data, state)
        ranked = []
        for inst, config in enumerate(_o256_CONFIGS):
            try:
                attr = _o256_launch(x, plan, query=1, inst=inst)
                if attr < 0 or attr % 10000:
                    print('v243 O256 config rejected resources', inst, config,
                          attr, flush=True)
                    continue
                candidate = lambda inst=inst: _o256_launch(x, plan, inst=inst)
                got = candidate()
                torch.cuda.synchronize(w.device)
                if not bool(torch.isfinite(got).all()):
                    continue
                cand_ms, base_ms = _v102_paired_graph_ms(
                    candidate, incumbent, w.device, repeats=13)
                cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
                core_ratio = (max(cand_us - 4.0, 1e-6) /
                              max(base_us - 4.0, 1e-6))
                ranked.append((core_ratio, cand_us, base_us, inst, attr // 10000))
                print('v243 O256 config', inst, config, 'candidate_us', cand_us,
                      'incumbent_us', base_us, 'core_ratio', core_ratio,
                      'regs', attr // 10000, flush=True)
            except Exception as exc:
                print('v243 O256 config unavailable', inst, config,
                      type(exc).__name__, str(exc)[:220], flush=True)
        distinct = [row for row in ranked if row[3] != 0]
        if not distinct:
            return state
        best = min(distinct)
        ratio, cand_us, base_us, inst, regs = best
        valid = ratio <= 0.99 and cand_us < base_us
        if valid:
            valid = _o256_validate(plan, w, inst)
        plan['inst'] = inst
        plan['enabled'] = bool(valid)
        print('v243 O256 decision', inst, _o256_CONFIGS[inst],
              'candidate_us', cand_us, 'incumbent_us', base_us,
              'core_ratio', ratio, 'regs', regs,
              'enabled', plan['enabled'], flush=True)
    except Exception as exc:
        state['_v243_o256'] = {'enabled': False}
        print('v243 O256 unavailable', type(exc).__name__, str(exc)[:300],
              flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    plan = state.get('_v243_o256', {})
    if ((cfg['layer'], cfg['m'], cfg['n'], cfg['k']) ==
            ('o', 256, 8192, 1024) and plan.get('enabled')):
        return _o256_launch(data['x'], plan, inst=plan['inst'])
    return _v243_base_kernel_fn(data, state)
'''

text = base + "\n\n\n" + support + wrapper.rstrip() + "\n"
compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
