"""Generate a guarded QKV m=1 Codec E GEMV candidate from protected v105."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_cgemv_threads_v105.py"
CODEC = ROOT / "codec_e.py"
GEMV = ROOT / "codec_e_gemv_probe.py"
OUT = ROOT / "submission_codec_e_gemv_v148.py"


base = BASE.read_text()
codec = CODEC.read_text().replace("from __future__ import annotations\n", "")
gemv = GEMV.read_text()
gemv_runtime = gemv[gemv.index('CUDA_SOURCE = r"""'):]


integration = r'''

# ---------------------------------------------------------------- v148: guarded QKV m=1 Codec E GEMV.
# Four independent warp-interleaved entropy streams recreate the existing C12
# high-code fragment.  dec1, exact escapes, MMA order and reduction are kept.
import threading as _v148_threading

_V148_REPLAY_OVERHEAD_US = 4.0
_V148_ROUTE_CORE_RATIO = 0.97
_v148_base_prepare = prepare
_v148_base_kernel_fn = kernel_fn


def _v148_validate(exp, weight):
    exact_pack = verify_codec_e_gemv_pack(exp)
    resources = {stage: codec_e_gemv_resource_info(stage) for stage in (1, 2)}
    generator = torch.Generator(device=weight.device).manual_seed(148001)
    x = torch.randn((1, 8192), generator=generator, dtype=torch.float32,
                    device=weight.device).to(torch.bfloat16)
    random_results = {}
    for stage in (1, 2):
        output = run_codec_e_gemv(exp, x, stages=stage)
        reference = (x.double() @ weight.double().t()).to(torch.bfloat16).double()
        torch.cuda.synchronize(weight.device)
        relative = float((output.double() - reference).norm() / reference.norm())
        random_results[stage] = {
            'finite': bool(torch.isfinite(output).all()),
            'relative': relative,
        }

    one_hot = True
    probe = torch.zeros((1, 8192), dtype=torch.bfloat16, device=weight.device)
    for column in (0, 63, 64, 2731, 5461, 8191):
        probe.zero_()
        probe[0, column] = 1
        for stage in (1, 2):
            got = run_codec_e_gemv(exp, probe, stages=stage)
            one_hot = one_hot and bool(torch.equal(got[0], weight[:, column]))
    torch.cuda.synchronize(weight.device)
    valid = exact_pack and one_hot
    for stage in (1, 2):
        valid = (valid and random_results[stage]['finite'] and
                 random_results[stage]['relative'] < 5e-3 and
                 resources[stage]['local_bytes'] == 0 and
                 resources[stage]['active_ctas_per_sm'] >= 1)
    print('v148 Codec E validation exact_pack', exact_pack, 'one_hot', one_hot,
          'random', random_results, 'resources', resources,
          'bits_per_weight', exp['bits_per_weight'], flush=True)
    return valid, resources


def prepare(weights):
    weight = weights.get('qkv_gate')
    h200 = (weight is not None and weight.shape == (2304, 8192) and
            weight.dtype == torch.bfloat16 and weight.is_cuda and
            torch.cuda.get_device_capability(weight.device) == (9, 0) and
            'H200' in torch.cuda.get_device_name(weight.device))
    build = None
    if h200:
        build = _v148_threading.Thread(target=e_gemv_extension)
        build.start()
    state = _v148_base_prepare(weights)
    state['_v148_codec_e'] = {'enabled': False}
    if not h200:
        return state
    if build is not None:
        build.join()
    try:
        exp = prepare_codec_e_gemv(weight)
        valid, resources = _v148_validate(exp, weight)
        for key in tuple(exp):
            if key.startswith('_debug_'):
                exp.pop(key)
        if not valid:
            print('v148 Codec E rejected: exactness, accuracy, spills, or residency', flush=True)
            return state

        generator = torch.Generator(device=weight.device).manual_seed(148101)
        x = torch.randn((1, 8192), generator=generator, dtype=torch.float32,
                        device=weight.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'm': 1, 'n': 2304, 'k': 8192, 'layer': 'qkv_gate'}}
        timings = []
        for stage in (1, 2):
            candidate_ms, incumbent_ms = _v102_paired_graph_ms(
                lambda stage=stage: run_codec_e_gemv(exp, x, stages=stage),
                lambda: _v148_base_kernel_fn(data, state),
                weight.device, repeats=13)
            candidate_us = candidate_ms * 1e3
            incumbent_us = incumbent_ms * 1e3
            candidate_core = max(candidate_us - _V148_REPLAY_OVERHEAD_US, 0.05)
            incumbent_core = max(incumbent_us - _V148_REPLAY_OVERHEAD_US, 0.05)
            ratio = candidate_core / incumbent_core
            timings.append((ratio, candidate_us, incumbent_us, stage))
            print('v148 Codec E qkv m1 stage', stage,
                  'candidate_us', candidate_us, 'incumbent_us', incumbent_us,
                  'core_ratio', ratio, 'resources', resources[stage], flush=True)
        timings.sort()
        ratio, candidate_us, incumbent_us, stage = timings[0]
        enabled = candidate_us < incumbent_us and ratio <= _V148_ROUTE_CORE_RATIO
        state['_v148_codec_e'] = {
            'enabled': enabled,
            'stage': stage,
            'state': exp,
            'core_ratio': ratio,
            'candidate_us': candidate_us,
            'incumbent_us': incumbent_us,
        }
        print('v148 Codec E qkv m1 selected_stage', stage,
              'enabled', enabled, 'core_ratio', ratio, flush=True)
    except Exception as exc:
        print('v148 Codec E unavailable', type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    config = data['config']
    exp = state.get('_v148_codec_e')
    if (exp is not None and exp.get('enabled') and
            config['layer'] == 'qkv_gate' and config['m'] == 1 and
            config['n'] == 2304 and config['k'] == 8192):
        x = data['x']
        codec_state = exp['state']
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == codec_state['low'].device and x.data_ptr() % 16 == 0):
            return run_codec_e_gemv(codec_state, x, stages=exp['stage'])
    return _v148_base_kernel_fn(data, state)
'''


OUT.write_text(base + "\n\n" + codec + "\n\nimport hashlib\n\n" + gemv_runtime + integration)
print(OUT)
