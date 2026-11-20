"""Generate v274: pair planes with exact predecoded FP32 escape deltas."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_plain_pairplanes_m1_margin_v248.py"
OUTPUT = ROOT / "submission_plain_pairplanes_predelta_v274.py"

text = BASE.read_text(encoding="utf-8").rstrip()
tree = ast.parse(text)
cuda = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        if any(isinstance(t, ast.Name) and t.id == "_V106_CG_CUDA"
               for t in node.targets):
            cuda = ast.literal_eval(node.value)
            break
if not isinstance(cuda, str):
    raise RuntimeError("v274 donor CUDA marker mismatch")
cuda = cuda.replace("cgemv_pair", "cgemv_pair_delta")
old_decode = "esc_delta((uint32_t)en.y, base4 & 0xFFu)"
new_decode = "__uint_as_float((uint32_t)en.y)"
if cuda.count(old_decode) != 1:
    raise RuntimeError("v274 escape decode marker mismatch")
cuda = cuda.replace(old_decode, new_decode, 1)

text += "\n\n\n# v274: exact escape deltas are materialized once during prepare.\n"
text += "_V274_CUDA = r'''\n" + cuda.rstrip() + "\n'''\n"
text += r'''

_V274_SHAPES = {
    'qkv_gate': (2304, 8192, 512),
    'o': (8192, 1024, 64),
    'pooler': (4096, 8192, 512),
    'unpooler': (1024, 4096, 128),
}
_V274_REPLAY_OVERHEAD_US = 4.0
_V274_REQUIRED_CORE_RATIO = 0.99

RGEMV_SRC += _V274_CUDA
CPP_SRC += r"""
void cgemv_pair_delta(torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
                      torch::Tensor, int64_t, int64_t);
int64_t cgemv_pair_delta_info(int64_t);
"""


def _get_ext():
    """v274 translation unit with literal predecoded-delta pair kernel."""
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace(
            "@@BODY@@", WG_HELPERS if sm90 else "")
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC +
                    _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90) +
                    _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC +
                    _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v274_predelta_{arch}" +
                 (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq",
                       "gqx", "rgemv", "sgemv", "cgemv", "cgemv_info",
                       "cgemv_pair", "cgemv_pair_info",
                       "cgemv_pair_delta", "cgemv_pair_delta_info", "rk"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                f"-gencode=arch=compute_{arch},code=sm_{arch}"
            ] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


def _v274_delta_table(cg):
    """Replace stored BF16 escape bits by the identical FP32 delta bits."""
    tab = cg['tab'].clone()
    bits = tab[:, :, 1]
    base = int(cg['base4']) & 0xFF
    hi = (bits >> 8) & 0xFF
    e71 = hi & 0x7F
    dhi = (hi & 0x80) | torch.where(
        e71 < base, torch.full_like(e71, base),
        torch.full_like(e71, base + 7))
    dbits = ((dhi << 8) | (bits & 0xFF)) & 0xFFFF
    original = ((bits & 0xFFFF).to(torch.int16).contiguous()
                .view(torch.bfloat16).float())
    decoded = (dbits.to(torch.int16).contiguous()
               .view(torch.bfloat16).float())
    delta = (original - decoded).contiguous()
    count = tab[:, 0, 0]
    slot = torch.arange(tab.shape[1], device=tab.device).view(1, -1)
    used = (slot >= 1) & (slot <= count.view(-1, 1))
    exact = bool(torch.equal((decoded + delta)[used], original[used]))
    tab[:, :, 1] = delta.view(torch.int32)
    return tab.contiguous(), exact


def _v274_run(x, layer, n, state):
    entry = state['_v274_predelta'][layer]
    cg = state['cg'][layer]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().cgemv_pair_delta(
        entry['packed'], x, entry['table'], cg['pmax'], out,
        cg['base4'], entry['nth'])
    return out


def _v274_resource_info(k, nth):
    packed = int(_get_ext().cgemv_pair_delta_info(_V106_INFO_IDS[(k, nth)]))
    return {
        'registers': packed & ((1 << 10) - 1),
        'local_bytes': (packed >> 10) & ((1 << 20) - 1),
        'static_shared_bytes': (packed >> 30) & ((1 << 20) - 1),
        'active_ctas_per_sm': packed >> 50,
    }


_v274_base_prepare = prepare
_v274_base_kernel_fn = kernel_fn


def prepare(weights):
    state = _v274_base_prepare(weights)
    wq = weights.get('qkv_gate')
    if (wq is None or not wq.is_cuda or
            torch.cuda.get_device_capability(wq.device) != (9, 0) or
            'H200' not in torch.cuda.get_device_name(wq.device)):
        return state
    state['_v274_predelta'] = {}
    for li, (layer, (n, k, nth)) in enumerate(_V274_SHAPES.items()):
        try:
            if layer not in state.get('cg', {}):
                continue
            pair, pack_exact = _v106_pair_pack(state['cg'][layer])
            table, table_exact = _v274_delta_table(state['cg'][layer])
            state['_v274_predelta'][layer] = {
                'packed': pair, 'table': table, 'nth': nth,
                'enabled': False, 'pack_exact': pack_exact,
                'table_exact': table_exact,
            }
            resources = _v274_resource_info(k, nth)
            gen = torch.Generator(device=wq.device).manual_seed(274101 + li)
            x = torch.randn((1, k), generator=gen, dtype=torch.float32,
                            device=wq.device).to(torch.bfloat16)
            x[0, 0] = 0
            x[0, k - 1] = 16
            data = {'x': x, 'config': {
                'm': 1, 'n': n, 'k': k, 'layer': layer}}
            incumbent = _v274_base_kernel_fn(data, state)
            candidate = _v274_run(x, layer, n, state)
            torch.cuda.synchronize(wq.device)
            bitwise = bool(torch.equal(candidate, incumbent))
            w = weights[layer][:n]
            ref = (x.double() @ w.double().t()).to(torch.bfloat16).double()
            rel = float(((candidate.double() - ref).norm() /
                         ref.norm()).item())
            finite = bool(torch.isfinite(candidate).all())
            probe_exact = True
            one = torch.zeros((1, k), dtype=torch.bfloat16, device=wq.device)
            for col in (0, k // 3, (2 * k) // 3, k - 1):
                one.zero_()
                one[0, col] = 1
                got = _v274_run(one, layer, n, state)
                probe_exact = (probe_exact and
                               bool(torch.equal(got[0], w[:, col])))
            torch.cuda.synchronize(wq.device)
            cand_ms, base_ms = _v102_paired_graph_ms(
                lambda: _v274_run(x, layer, n, state),
                lambda: _v274_base_kernel_fn(data, state),
                wq.device, repeats=13)
            cand_us, base_us = cand_ms * 1e3, base_ms * 1e3
            cand_core = max(cand_us - _V274_REPLAY_OVERHEAD_US, 1e-6)
            base_core = max(base_us - _V274_REPLAY_OVERHEAD_US, 1e-6)
            core_ratio = cand_core / base_core
            valid = (pack_exact and table_exact and bitwise and probe_exact and
                     finite and rel < 5e-3 and resources['local_bytes'] == 0 and
                     resources['active_ctas_per_sm'] >= 1)
            enable = (valid and cand_us < base_us and
                      core_ratio <= _V274_REQUIRED_CORE_RATIO)
            state['_v274_predelta'][layer].update({
                'enabled': enable, 'core_ratio': core_ratio})
            print('v274 pair predelta', layer, 'resources', resources,
                  'pack_exact', pack_exact, 'table_exact', table_exact,
                  'bitwise_base', bitwise, 'probe_exact', probe_exact,
                  'finite', finite, 'fp64_rel', rel,
                  'candidate_us', cand_us, 'incumbent_us', base_us,
                  'core_ratio', core_ratio, 'enabled', enable, flush=True)
        except Exception as exc:
            print('v274 pair predelta unavailable', layer,
                  type(exc).__name__, str(exc), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    entry = state.get('_v274_predelta', {}).get(cfg['layer'])
    expected = _V274_SHAPES.get(cfg['layer'])
    if (entry is not None and entry.get('enabled') and cfg['m'] == 1 and
            expected is not None and (cfg['n'], cfg['k']) == expected[:2]):
        x = data['x']
        packed = entry['packed']
        if (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and
                x.device == packed.device and x.data_ptr() % 16 == 0):
            return _v274_run(x, cfg['layer'], cfg['n'], state)
    return _v274_base_kernel_fn(data, state)
'''

compile(text, str(OUTPUT), "exec")
with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
    stream.write(text)
print(OUTPUT.name, len(text.encode("utf-8")))
