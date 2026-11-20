import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.utils.cpp_extension as ce
import torch

# The development GPU is Ampere; all candidates in this helper are Hopper-only.
torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)

_load_inline = ce.load_inline


def _compat_load_inline(*args, **kwargs):
    kwargs.pop("no_implicit_headers", None)
    return _load_inline(*args, **kwargs)


ce.load_inline = _compat_load_inline

module_name = sys.argv[1] if len(sys.argv) > 1 else "submission_s094_route_finalizer"
candidate = importlib.import_module(module_name)

if "s100" in module_name:
    print(candidate._att7())
if hasattr(candidate, "_s094_route_ext"):
    print(candidate._s094_route_ext())
if hasattr(candidate, "_s097_ext"):
    print(candidate._s097_ext())
if hasattr(candidate, "_S096_M16_DEFS"):
    print(candidate._ma_ext(candidate._S096_M16_DEFS))
if hasattr(candidate, "_S098_M16_DEFS"):
    print(candidate._ma_ext(candidate._S098_M16_DEFS))
if hasattr(candidate, "_S099_M32_DEFS"):
    print(candidate._ma_ext(candidate._S099_M32_DEFS))
if hasattr(candidate, "_S099_SK2_CONFIGS"):
    print(candidate.bfp__get_ext())
