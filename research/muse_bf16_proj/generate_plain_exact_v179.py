"""Generate a readable v150 submission with the exact live Nathen v18 routes.

Only the definitions reachable from the eleven v105 alternate routes are
vendored.  Encoded CUDA strings are expanded by this offline generator, while
dead historical experiments are omitted to keep the single-file submission
below the official size limit.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import inspect
import sys
import textwrap
import types
import zlib
from pathlib import Path
from types import CodeType


ROOT = Path(__file__).resolve().parent
BASE = ROOT / (sys.argv[1] if len(sys.argv) > 1
               else "submission_rs128_faithful_v150.py")
NATHEN = ROOT / "submission_nathen_v18.py"
OUTPUT = ROOT / (sys.argv[2] if len(sys.argv) > 2
                 else "submission_plain_exact_v180.py")
STRIP_MARKER = sys.argv[3] if len(sys.argv) > 3 else None


def raw_literal(value: str) -> str:
    for quote in ('"""', "'''"):
        if quote not in value and not value.endswith("\\"):
            return "r" + quote + value + quote
    raise ValueError("source cannot be represented by one raw literal")


def readable_line_literal(value: str) -> str:
    """Preserve native source bytes when the assignment is later indented.

    Adjacent ordinary string literals keep each CUDA line visible in the
    submitted file without letting lexical namespace indentation become part
    of the string value.
    """
    chunks = value.splitlines(keepends=True)
    if not chunks:
        return "''"
    return "(\n" + "".join(f"    {chunk!r}\n" for chunk in chunks) + ")"


def assigned_literal(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(name)
            return value
    raise KeyError(name)


def bound_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name):
                    names.add(child.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    return names


def code_global_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(code_global_names(value))
    return names


def literal_container(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(literal_container(item) for item in value)
    if isinstance(value, dict):
        return all(literal_container(k) and literal_container(v)
                   for k, v in value.items())
    return False


def load_nathen_module():
    # The dependency slicer never calls the SGLang fallback, but the historical
    # source imports it at module load.  A local stub keeps generation offline.
    module_names = (
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[module_names[-1]]._bf16_gemm_dispatch_impl = (
            lambda *args, **kwargs: None
        )
        spec = importlib.util.spec_from_file_location("_nathen_v18_source", NATHEN)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


source = NATHEN.read_text()
tree = ast.parse(source)
module = load_nathen_module()
namespace = vars(module)

binding: dict[str, ast.AST] = {}
function_at_line: dict[int, ast.AST] = {}
for node in tree.body:
    for name in bound_names(node):
        binding[name] = node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        function_at_line[node.lineno] = node

seeds = {
    "prepare_router", "run_router",
    "prepare_olayer_candidates", "run_olayer_candidate",
    "tt_v10_sm2_prepare_smallm2", "tt_v10_sm2_run_smallm2",
    "prepare_v10_tune", "run_v10_tune",
    "tt_v13_smtrace8_prepare_smtrace", "tt_v13_smtrace8_run_smtrace",
    "tt_v14_qkv_prepare_qkvbig", "tt_v14_qkv_run_qkvbig",
    "tt_v14_qkv_QKVBIG_CUDA", "tt_v14_qkv_qbig_mma",
    "tt_v14_qkv_qbig_pack",
    "dd_prepare_decodediet", "dd_run_decodediet",
    "rt_prepare_routertail_candidates", "rt_run_routertail_candidate",
    "v18s_build", "v18s_plan", "v18s_bf16_twin", "v18s_launch_raw",
    "V18_SMALL",
}

external = {
    "torch", "hashlib", "load_inline", "TypedDict", "F",
    "_zsrc", "_zsrc_b64", "_zsrc_zlib", "base64", "zlib",
}
selected: dict[int, ast.AST] = {}
pending = list(seeds)
seen: set[str] = set()

while pending:
    name = pending.pop()
    if name in seen or name in external:
        continue
    seen.add(name)
    if name not in namespace:
        raise RuntimeError(f"missing runtime dependency {name}")
    value = namespace[name]
    node = None
    deps: set[str] = set()

    if inspect.isfunction(value) and Path(inspect.getsourcefile(value) or "") == NATHEN:
        node = function_at_line.get(value.__code__.co_firstlineno)
        if node is None:
            raise RuntimeError(f"no AST node for {name} at {value.__code__.co_firstlineno}")
        deps.update(code_global_names(value.__code__))
        # Defaults, decorators and annotations are evaluated when the function
        # is defined, outside its code object.
        definition_bits = [*node.decorator_list, *node.args.defaults,
                           *node.args.kw_defaults]
        if node.returns is not None:
            definition_bits.append(node.returns)
        definition_bits.extend(
            arg.annotation for arg in [*node.args.posonlyargs, *node.args.args,
                                       *node.args.kwonlyargs]
            if arg.annotation is not None
        )
        for bit in definition_bits:
            deps.update(child.id for child in ast.walk(bit)
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load))
    elif inspect.isclass(value) and value.__module__ == module.__name__:
        node = binding.get(name)
        if node is not None:
            deps.update(child.id for child in ast.walk(node)
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load))
    else:
        node = binding.get(name)
        if node is not None:
            deps.update(child.id for child in ast.walk(node)
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load))

    if node is None:
        # Imported modules/functions are supplied by the slim prelude.
        if inspect.ismodule(value) or inspect.isbuiltin(value) or inspect.isfunction(value):
            continue
        raise RuntimeError(f"no source binding for dependency {name}: {type(value)}")

    selected[id(node)] = node
    pending.extend(dep for dep in deps if dep in namespace and dep not in seen)


lines = source.splitlines(keepends=True)
parts = [
    "from typing import TypedDict\n",
    "import hashlib\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.cpp_extension import load_inline\n\n",
]

for node in sorted(selected.values(), key=lambda item: item.lineno):
    names = bound_names(node)
    segment = "".join(lines[node.lineno - 1:node.end_lineno])
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and len(names) == 1:
        name = next(iter(names))
        value = namespace[name]
        if isinstance(value, str) and "_zsrc(" in segment:
            segment = f"{name} = {readable_line_literal(value)}\n"
        elif literal_container(value):
            segment = f"{name} = {value!r}\n"
    parts.append(segment)
    if not segment.endswith("\n"):
        parts.append("\n")
    parts.append("\n")

parts.append(r'''
_SLIM_OLAYER_CONFIGS = [(128, 128, 4, 3, 1, 1)]
_SLIM_V18_O8 = V18_SMALL[('o', 8)]


def prepare(weights):
    state = {'weights': weights}
    state['router32'] = prepare_router({'router': weights['router']})
    state['router_tail'] = rt_prepare_routertail_candidates(
        {'router': weights['router']})

    state['olayer'] = prepare_olayer_candidates(
        {'o': weights['o']}, _SLIM_OLAYER_CONFIGS)
    x = torch.zeros((128, 1024), dtype=torch.bfloat16,
                    device=weights['o'].device)
    run_olayer_candidate(x, state['olayer'], 0)

    state['o256'] = prepare_v10_tune({'o': weights['o']})
    x = torch.zeros((256, 1024), dtype=torch.bfloat16,
                    device=weights['o'].device)
    run_v10_tune(x, 'o', 8192, state['o256'])

    state['pooler8'] = tt_v10_sm2_prepare_smallm2(
        {'pooler': weights['pooler']})
    state['unpooler8'] = tt_v13_smtrace8_prepare_smtrace(
        {'unpooler': weights['unpooler']})
    state['qkv128'] = tt_v14_qkv_prepare_qkvbig(
        {'qkv_gate': weights['qkv_gate']})
    state['dense16'] = dd_prepare_decodediet({
        'pooler': weights['pooler'], 'unpooler': weights['unpooler']})

    inst, cfg = _SLIM_V18_O8
    instances = list(dict.fromkeys((inst, v18s_bf16_twin(inst))))
    ext = v18s_build(instances, 'v18_small_selected')
    ids = {value: index for index, value in enumerate(instances)}
    plan = v18s_plan(ext, ids, {}, {}, weights['o'], 'o', ('o', 'm', 8),
                     inst, cfg)
    if plan is None:
        twin = v18s_bf16_twin(inst)
        plan = v18s_plan(ext, ids, {}, {}, weights['o'], 'o',
                         ('o', 'm', 8), twin, cfg)
    state['o8'] = {'ext': ext, 'plan': plan}
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def _fallback(data, state):
    x, cfg = data['x'], data['config']
    weight = state['weights'][cfg['layer']]
    if cfg['layer'] != 'router':
        return x @ weight[:cfg['n']].T
    if x.shape[0] <= 8:
        return F.linear(x.float(), weight)
    hi = weight.to(torch.bfloat16)
    out = torch.mm(x, hi.T, out_dtype=torch.float32)
    out += torch.mm(x, (weight - hi.float()).to(torch.bfloat16).T,
                    out_dtype=torch.float32)
    return out


def kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    layer, m, n = cfg['layer'], cfg['m'], cfg['n']
    out = None
    if layer == 'o' and m == 8:
        plan = state['o8']['plan']
        if plan is not None:
            out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
            rc = v18s_launch_raw(state['o8']['ext'], plan['inst_id'], plan,
                                 x, out, n, plan['cfg'])
            if rc:
                out = None
    elif layer == 'o' and m == 128:
        out = run_olayer_candidate(x, state['olayer'], 0)
    elif layer == 'o' and m == 256:
        out = run_v10_tune(x, layer, n, state['o256'])
    elif layer == 'pooler' and m == 8:
        out = tt_v10_sm2_run_smallm2(x, layer, n, state['pooler8'])
    elif layer == 'unpooler' and m == 8:
        out = tt_v13_smtrace8_run_smtrace(x, layer, n, state['unpooler8'])
    elif layer in ('pooler', 'unpooler') and m == 16:
        out = dd_run_decodediet(x, layer, n, state['dense16'])
    elif layer == 'qkv_gate' and m == 128:
        out = tt_v14_qkv_run_qkvbig(x, layer, n, state['qkv128'])
    elif layer == 'router' and m in (8, 16):
        out = rt_run_routertail_candidate(x, state['router_tail'])
    elif layer == 'router' and m == 32:
        out = run_router(x, layer, n, state['router32'])
    return _fallback(data, state) if out is None else out
''')

slim_alt = "".join(parts)
ast.parse(slim_alt)
compile(slim_alt, "<slim_nathen_v18>", "exec")

base = BASE.read_text()
if STRIP_MARKER is not None:
    if base.count(STRIP_MARKER) != 1:
        raise RuntimeError(
            f"expected one strip marker, found {base.count(STRIP_MARKER)}: "
            f"{STRIP_MARKER!r}")
    base = base[:base.index(STRIP_MARKER)].rstrip() + "\n"
loader_start = "import types as _types, zlib as _zl, base64 as _b64\n"
loader_end = (
    'exec(compile(_zl.decompress(_b64.b64decode(_ALT_SRC)).decode(), '
    '"_alt_proj", "exec"), _alt.__dict__)\n'
)
if base.count(loader_start) != 1 or base.count(loader_end) != 1:
    raise RuntimeError("unexpected v150 alternate loader")
start = base.index(loader_start)
end = base.index(loader_end, start) + len(loader_end)
block = base[start:end]
route_line = next(line for line in block.splitlines()
                  if line.startswith("_ALT_ROUTES = "))
replacement = (
    "import types as _types\n" + route_line + "\n\n"
    "def _v179_build_alt_namespace():\n"
    + textwrap.indent(slim_alt, "    ")
    + "\n    return _types.SimpleNamespace(**locals())\n\n"
    "_alt = _v179_build_alt_namespace()\n"
    "del _v179_build_alt_namespace\n"
)
base = base[:start] + replacement + base[end:]
base = base.replace("_ALT_SRC", "readable alternate source")

payload = assigned_literal(base, "_V150_CUDA_PAYLOAD")
cuda = zlib.decompress(base64.b64decode(payload)).decode()
payload_start = base.index('_V150_CUDA_PAYLOAD = "')
payload_end = base.index("\n", payload_start) + 1
base = (base[:payload_start]
        + "_V150_CUDA_SOURCE = " + raw_literal(cuda) + "\n"
        + base[payload_end:])
base = base.replace("import base64 as _v150_base64\n", "", 1)
base = base.replace("import zlib as _v150_zlib\n", "", 1)
old = (
    "def _v150_cuda_source():\n"
    "    return _v150_zlib.decompress("
    "_v150_base64.b64decode(_V150_CUDA_PAYLOAD)).decode()\n"
)
new = "def _v150_cuda_source():\n    return _V150_CUDA_SOURCE\n"
if base.count(old) != 1:
    raise RuntimeError("unexpected v150 CUDA decoder")
base = base.replace(old, new, 1)

for forbidden in (
    "_ALT_SRC", "_V150_CUDA_PAYLOAD", "b64decode", "decompress(",
    "exec(", "compile(", "_zsrc", "base64",
):
    if forbidden in base:
        position = base.index(forbidden)
        context = base[max(0, position - 120):position + len(forbidden) + 120]
        raise RuntimeError(
            f"generated submission contains {forbidden!r}: {context!r}")

encoded_size = len(base.encode())
if encoded_size > 1_000_000:
    raise RuntimeError(f"generated submission exceeds 1 MB: {encoded_size}")
ast.parse(base)
compile(base, str(OUTPUT), "exec")
OUTPUT.write_text(base)
print(OUTPUT.name, encoded_size, "selected_nodes", len(selected),
      "slim_alt_bytes", len(slim_alt.encode()))
