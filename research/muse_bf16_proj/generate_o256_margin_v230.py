"""Generate v230: readable O/m256 Hopper margin isolate on v221.

The historical v56 kernel measured 12.640 versus 12.896 us but was rejected
by its old 3% gate.  This generator expands that one CUDA payload offline and
writes literal CUDA into the submission.  The submitted file performs no
source decoding and retains v221 as an immediate fallback.
"""

from pathlib import Path
import base64
import re
import zlib


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_full_guard_v221.py"
DONOR = ROOT / "submission_o256_v56.py"
OUTPUT = ROOT / "submission_plain_o256_margin_v230.py"

text = BASE.read_text(encoding="utf-8")
donor = DONOR.read_text(encoding="utf-8")
match = re.search(r'_o256_RL_CUDA = _q128_zsrc\("([A-Za-z0-9+/=]+)"\)', donor)
if match is None:
    raise RuntimeError("v56 O256 CUDA payload marker mismatch")
cuda = zlib.decompress(base64.b64decode(match.group(1))).decode("utf-8")
if "'''" in cuda:
    raise RuntimeError("unexpected triple quote in O256 CUDA")

support_start = donor.index("_o256_RL_CPP =")
support_end = donor.index("\n\n\ndef _c11_one", support_start)
support = donor[support_start:support_end]
support = support.replace("ratio < 0.97", "ratio < 0.99")

wrapper = r'''

# v230: independently validate and time the readable O/m256 kernel against
# the complete v221 composition.  Failure or a sub-1% adjusted win retains
# the protected route.
_v230_base_prepare = prepare
_v230_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v230_base_prepare(weights)
    w = weights.get('o')
    if (w is None or not w.is_cuda or
            torch.cuda.get_device_capability(w.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(w.device)):
        return state
    try:
        plan = _o256_prepare(weights)
        state['_v230_o256'] = plan
        if not plan.get('valid'):
            return state
        gen = torch.Generator(device=w.device).manual_seed(230256)
        x = torch.randn((256, 1024), generator=gen, dtype=torch.float32,
                        device=w.device).to(torch.bfloat16)
        data = {'x': x, 'config': {
            'layer': 'o', 'm': 256, 'n': 8192, 'k': 1024}}
        candidate = _o256_launch(x, plan)
        incumbent = _v230_base_kernel_fn(data, state)
        torch.cuda.synchronize(w.device)
        finite = bool(torch.isfinite(candidate).all())
        cand_ms, base_ms = _v102_paired_graph_ms(
            lambda: _o256_launch(x, plan),
            lambda: _v230_base_kernel_fn(data, state),
            w.device, repeats=13)
        cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
        cand_core = max(cand_us - 4.0, 1e-6)
        base_core = max(base_us - 4.0, 1e-6)
        ratio = cand_core / base_core
        plan['enabled'] = finite and cand_us < base_us and ratio <= 0.99
        print('v230 O256 margin candidate_us', cand_us,
              'incumbent_us', base_us, 'core_ratio', ratio,
              'finite', finite, 'enabled', plan['enabled'], flush=True)
    except Exception as exc:
        state['_v230_o256'] = {'enabled': False}
        print('v230 O256 margin unavailable', type(exc).__name__,
              str(exc)[:300], flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    plan = state.get('_v230_o256', {})
    if (cfg['layer'], cfg['m'], cfg['n'], cfg['k']) == ('o', 256, 8192, 1024):
        out = _o256_run(data['x'], cfg['layer'], cfg['n'], plan)
        if out is not None:
            return out
    return _v230_base_kernel_fn(data, state)
'''

block = "_o256_RL_CUDA = r'''" + cuda + "'''\n" + support + wrapper
text = text.rstrip() + "\n\n\n" + block.rstrip() + "\n"
OUTPUT.write_text(text, encoding="utf-8", newline="\n")
compile(text, str(OUTPUT), "exec")
print(OUTPUT.name, len(text.encode("utf-8")))
