"""Compile a v298-style submission's exact extension for Hopper without launching it."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import torch
from torch.utils import cpp_extension


if len(sys.argv) != 2:
    raise SystemExit("usage: compile_v298_sm90.py SUBMISSION.py")

source = pathlib.Path(sys.argv[1]).resolve()
module_name = f"_okc_compile_{source.stem}"
spec = importlib.util.spec_from_file_location(module_name, source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {source}")

original_load_inline = cpp_extension.load_inline


def load_inline_verbose(*args, **kwargs):
    flags = list(kwargs.get("extra_cuda_cflags", ()))
    flags.append("--ptxas-options=-v")
    kwargs["extra_cuda_cflags"] = flags
    return original_load_inline(*args, **kwargs)


# The source selects its Hopper code and gencode from this query.  Loading the
# resulting extension on the local Ampere GPU is safe; this script never calls
# a kernel.
cpp_extension.load_inline = load_inline_verbose
torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._module()
if hasattr(module, "_zh_join"):
    module._zh_join()
if hasattr(module, "_vp2_join"):
    module._vp2_join()
if hasattr(module, "_module_rmscancel"):
    module._module_rmscancel()
if hasattr(module, "_occ3_module"):
    module._occ3_module()
print(f"sm_90a exact extension compile OK: {source.name}")
