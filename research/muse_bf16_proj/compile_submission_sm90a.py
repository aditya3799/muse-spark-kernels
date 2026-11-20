"""Compile a submission's exact extension translation unit for SM90a locally.

The local PyTorch 2.6 loader predates the server's no_implicit_headers keyword,
so the wrapper removes only that loader keyword.  CUDA source and flags remain
the submission's own.  This script compiles but never launches a GPU kernel.
"""

import runpy
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline as local_load_inline


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: compile_submission_sm90a.py SUBMISSION.py")
    torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)
    namespace = runpy.run_path(sys.argv[1])

    def compatible_load_inline(**kwargs):
        kwargs.pop("no_implicit_headers", None)
        return local_load_inline(**kwargs)

    namespace["_get_ext"].__globals__["load_inline"] = compatible_load_inline
    started = time.time()
    namespace["_get_ext"]()
    print(f"FULL SM90A EXTENSION COMPILE OK {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
